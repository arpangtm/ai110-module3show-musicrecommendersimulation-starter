"""
VibeFinder agent: the full applied AI pipeline.

    request
      -> input guardrail          (empty? junk? injection text?)
      -> split negations          ("no country" becomes an exclusion)
      -> retrieve                 (situations, genre guide, song catalog)
      -> plan                     (Gemini if configured, else rules)
      -> profile guardrail        (unknown genre? out of range numbers?)
      -> recommend                (base scorer from src/recommender.py)
      -> self-critique            (confidence, mood/energy checks)
      -> revise if needed         (try a different weighting strategy)
      -> answer + trace           (explanations grounded in retrieved notes)

Every step is written to a trace so you can see what the agent decided and
why. Traces go to logs/agent_trace.jsonl, and normal logs to logs/vibefinder.log.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src import guardrails
from src.llm import LLMPlanner, gemini_generate_fn
from src.planner import Plan, RulePlanner, split_negations
from src.recommender import DEFAULT_WEIGHTS, load_songs, max_possible_score, recommend_songs
from src.retriever import Retriever, RetrievedDoc

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "songs.csv"
KNOWLEDGE_DIR = ROOT / "knowledge"
LOG_DIR = ROOT / "logs"

# Weighting strategies the critic can switch between.
STRATEGIES: Dict[str, Dict[str, float]] = {
    "balanced": dict(DEFAULT_WEIGHTS),
    "mood_first": {**DEFAULT_WEIGHTS, "mood": 3.0, "genre": 1.0},
    "energy_first": {**DEFAULT_WEIGHTS, "energy": 3.5, "genre": 1.0},
}

LOW_CONFIDENCE = 0.6
MIN_FIT = 0.2          # songs below this are not shown at all
THIN_REQUEST_CAP = 0.5 # max confidence when the profile has a single preference

ISSUE_TEXT = {
    "low_confidence": "Even the best matches only cover part of what you asked for.",
    "mood_missing_in_top3": "None of the top 3 have the mood you asked for; the songs with that mood don't fit your other preferences.",
    "energy_mismatch": "The top songs are not very close to the energy level you asked for.",
}
MAX_REVISIONS = 2


@dataclass
class Recommendation:
    rank: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    score: float
    fit: float
    reasons: str


@dataclass
class AgentResult:
    request: str
    ok: bool
    message: str = ""
    planner: str = ""
    strategy: str = ""
    profile: Dict = field(default_factory=dict)
    recommendations: List[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = ""
    grounding: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def configure_logging(log_dir: Path = LOG_DIR, verbose: bool = False) -> None:
    """File log at INFO, console only shows warnings unless verbose."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(getattr(h, "_vibefinder", False) for h in root.handlers):
        return
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir / "vibefinder.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh._vibefinder = True  # type: ignore[attr-defined]
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    ch._vibefinder = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    root.addHandler(ch)


class VibeFinderAgent:
    def __init__(
        self,
        songs: Optional[List[Dict]] = None,
        use_llm: bool = True,
        llm_generate: Optional[Callable[[str], str]] = None,
        trace_path: Optional[Path] = LOG_DIR / "agent_trace.jsonl",
    ):
        self.songs = songs if songs is not None else load_songs(str(DATA_PATH))
        self.genres = sorted({s["genre"] for s in self.songs})
        self.moods = sorted({s["mood"] for s in self.songs})
        self.retriever = Retriever(KNOWLEDGE_DIR, self.songs)
        genre_docs = [d for d in self.retriever.docs if d.source == "genres"]
        self.rule_planner = RulePlanner(genre_docs)

        self.llm_planner: Optional[LLMPlanner] = None
        if use_llm:
            generate = llm_generate or gemini_generate_fn()
            if generate:
                self.llm_planner = LLMPlanner(generate, self.genres, self.moods)
        self.trace_path = trace_path

    # -- helpers -----------------------------------------------------------

    def _plan(self, request: str, retrieved: List[RetrievedDoc], result: AgentResult) -> Plan:
        if self.llm_planner:
            try:
                plan = self.llm_planner.plan(request, retrieved)
                self._step(result, "plan", f"LLM planner returned {plan.profile}")
                return plan
            except Exception as exc:  # network, quota, bad JSON ...
                logger.warning("LLM planner failed (%s); falling back to rules", exc)
                result.warnings.append("The AI planner failed, so the rule-based planner was used instead.")
                self._step(result, "plan", f"LLM failed: {exc}. Falling back to rules.")
        plan = self.rule_planner.plan(request, retrieved)
        self._step(result, "plan", f"Rule planner built {plan.profile} from {plan.sources}")
        return plan

    @staticmethod
    def _step(result: AgentResult, name: str, detail) -> None:
        result.trace.append({"step": name, "detail": detail})

    def critique(self, profile: Dict, recs, weights: Dict[str, float], catalog: List[Dict]) -> Dict:
        """Grade a candidate list. Returns confidence, fit per song and issues."""
        best = max_possible_score(profile, weights) or 1.0
        fits = [max(0.0, score) / best for _, score, _ in recs]
        top = fits[:3]
        confidence = round(sum(top) / len(top), 3) if top else 0.0
        issues: List[str] = []

        if confidence < LOW_CONFIDENCE:
            issues.append("low_confidence")
        if profile.get("mood") and recs:
            if not any(s["mood"] == profile["mood"] for s, _, _ in recs[:3]):
                available = any(s["mood"] == profile["mood"] for s in catalog)
                if available:
                    issues.append("mood_missing_in_top3")
        if profile.get("energy") is not None and recs:
            gaps = [abs(s["energy"] - profile["energy"]) for s, _, _ in recs[:3]]
            if sum(gaps) / len(gaps) > 0.25:
                issues.append("energy_mismatch")
        return {"confidence": confidence, "fits": [round(f, 3) for f in fits], "issues": issues}

    # -- main loop ---------------------------------------------------------

    def run(self, raw_request: str) -> AgentResult:
        started = time.time()
        result = AgentResult(request=raw_request or "", ok=False)

        # 1. input guardrail
        check = guardrails.check_request(raw_request)
        result.warnings += check.warnings
        self._step(result, "check_input", {"ok": check.ok, "text": check.text, "reason": check.reason})
        if not check.ok:
            result.message = check.reason
            return self._finish(result, started)
        request = check.text

        # 2-3. negations + retrieval
        positive_text, negated = split_negations(request)
        retrieved = self.retriever.search(positive_text)
        self._step(result, "retrieve", {
            "query": positive_text,
            "negated": negated,
            "hits": [{"source": h.doc.source, "title": h.doc.title, "score": h.score,
                      "matched": h.matched} for h in retrieved],
        })

        # 4. plan
        plan = self._plan(request, retrieved, result)
        result.planner = plan.planner
        result.warnings += plan.notes

        # 5. profile guardrail
        profile, warnings = guardrails.validate_profile(plan.profile, self.genres, self.moods)
        result.warnings += warnings
        result.profile = profile
        self._step(result, "validate_profile", {"profile": profile, "warnings": warnings})

        if not guardrails.has_preferences(profile):
            result.message = (
                "I couldn't find anything about music, a mood, or an activity in that request, "
                "so I'm not going to guess. Try something like 'upbeat songs for a road trip'."
            )
            self._step(result, "refuse", "no usable preferences")
            return self._finish(result, started)

        # Build the candidate pool (exclusions + the reference song come out)
        excluded = set(profile.get("exclude_genres", []))
        catalog = [s for s in self.songs
                   if s["genre"] not in excluded and s["id"] != plan.seed_song_id]
        if profile.get("genre"):
            count = sum(1 for s in catalog if s["genre"] == profile["genre"])
            if count < profile["k"]:
                result.warnings.append(
                    f"Only {count} {profile['genre']} song(s) in the catalog, so the rest of the list "
                    f"comes from other genres that fit your mood/energy."
                )

        # 6-7. recommend, critique, revise
        attempts = []
        strategy = "balanced"
        tried = set()
        for round_no in range(MAX_REVISIONS + 1):
            tried.add(strategy)
            weights = STRATEGIES[strategy]
            recs = recommend_songs(profile, catalog, k=profile["k"], weights=weights)
            review = self.critique(profile, recs, weights, catalog)
            attempts.append((strategy, recs, review))
            self._step(result, "recommend" if round_no == 0 else "revise", {
                "strategy": strategy,
                "top": [s["title"] for s, _, _ in recs],
                "confidence": review["confidence"],
                "issues": review["issues"],
            })
            if not review["issues"]:
                break
            next_strategy = None
            if "mood_missing_in_top3" in review["issues"] and "mood_first" not in tried:
                next_strategy = "mood_first"
            elif "energy_mismatch" in review["issues"] and "energy_first" not in tried:
                next_strategy = "energy_first"
            if not next_strategy:
                break
            strategy = next_strategy

        # Pick the attempt with the fewest issues, then highest confidence.
        strategy, recs, review = min(
            attempts, key=lambda a: (len(a[2]["issues"]), -a[2]["confidence"])
        )
        self._step(result, "decide", {
            "chosen_strategy": strategy,
            "compared": {a[0]: {"confidence": a[2]["confidence"], "issues": a[2]["issues"]} for a in attempts},
        })

        if review["issues"] and len(attempts) > 1:
            result.warnings.append(
                "I tried re-weighting the ranking and couldn't fully satisfy the request, "
                "probably because the catalog has nothing that fits all of it at once."
            )
        for issue in review["issues"]:
            if issue != "low_confidence":
                result.warnings.append(ISSUE_TEXT[issue])

        # Output guardrail: don't pad the list with songs that match nothing.
        keep = [i for i, f in enumerate(review["fits"]) if f >= MIN_FIT] or [0]
        if len(keep) < len(recs):
            result.warnings.append(
                f"Only {len(keep)} song(s) were a reasonable match, so I left out "
                f"{len(recs) - len(keep)} weak filler pick(s)."
            )
        recs = [recs[i] for i in keep]
        review["fits"] = [review["fits"][i] for i in keep]

        # A request with only one usable preference ("not sad") will match lots
        # of songs perfectly on that one thing. That isn't real confidence.
        signals = sum(1 for key in ("genre", "mood", "energy", "likes_acoustic", "tempo_bpm", "valence")
                      if profile.get(key) is not None)
        if signals == 1 and review["confidence"] > THIN_REQUEST_CAP:
            review["confidence"] = THIN_REQUEST_CAP
            result.warnings.append(
                "Your request only gave me one thing to go on, so many songs tie. "
                "Add a genre, mood or activity for a sharper list."
            )

        result.strategy = strategy
        result.confidence = review["confidence"]
        result.confidence_label = (
            "high" if review["confidence"] >= 0.75 else "medium" if review["confidence"] >= 0.5 else "low"
        )
        if result.confidence_label == "low":
            result.warnings.append(
                "Low confidence: these are the closest songs in a 20-song catalog, not strong matches."
            )
        result.recommendations = [
            Recommendation(i, s["title"], s["artist"], s["genre"], s["mood"], s["energy"],
                           score, review["fits"][i - 1], why)
            for i, (s, score, why) in enumerate(recs, start=1)
        ]
        result.grounding = [
            f"{h.doc.title} ({h.doc.source}): {h.doc.text}" for h in retrieved
            if h.doc.source != "catalog"
        ][:3]
        result.ok = True
        return self._finish(result, started)

    def _finish(self, result: AgentResult, started: float) -> AgentResult:
        self._step(result, "done", {"ok": result.ok, "seconds": round(time.time() - started, 4)})
        logger.info("Request %r -> ok=%s strategy=%s confidence=%.2f",
                    result.request, result.ok, result.strategy, result.confidence)
        if self.trace_path:
            try:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "request": result.request, "trace": result.trace}) + "\n")
            except OSError as exc:
                logger.warning("Could not write trace: %s", exc)
        return result
