"""
Planner: turns a plain English request + retrieved context into a
preference profile the scorer understands.

Two planners share one interface (`plan(request, retrieved) -> Plan`):

  RulePlanner    always available, offline, deterministic. Uses the hints in
                 the retrieved documents, then layers explicit words from the
                 request on top ("slow", "acoustic", "top 3", "no country").
  LLMPlanner     (src/llm.py) asks Gemini to do the same job when an API key
                 is set. Its output still goes through the same guardrail.

Priority, lowest to highest:
  situation docs  <  a song the user named  <  genre docs  <  explicit words
So "rainy day but upbeat" keeps the rainy day mood and acoustic leaning, but
the word "upbeat" wins on energy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.retriever import RetrievedDoc, normalize

MOOD_WORDS = {
    "happy": ["happy", "upbeat", "cheerful", "joyful", "feel good"],
    "chill": ["chill", "chilled", "laid back"],
    "intense": ["intense", "aggressive", "powerful"],
    "relaxed": ["relaxed"],
    "moody": ["moody", "dark", "brooding"],
    "focused": ["focused"],
    "confident": ["confident", "swagger"],
    "euphoric": ["euphoric", "hype", "hyped"],
    "calm": ["calm", "peaceful", "soothing", "serene"],
    "nostalgic": ["nostalgic"],
    "angry": ["angry", "mad", "furious"],
    "romantic": ["romantic"],
    "melancholy": ["melancholy", "melancholic", "sad", "sadness", "bittersweet"],
}

ENERGY_WORDS = {
    0.88: ["high energy", "energetic", "hype", "hyped", "pump", "pumped", "fast", "loud", "banger", "bangers", "upbeat"],
    0.55: ["medium energy", "mid energy", "moderate"],
    0.22: ["low energy", "slow", "soft", "quiet", "mellow", "gentle", "sleepy", "lowkey", "low key"],
}

ACOUSTIC_TRUE = ["acoustic", "unplugged", "live instruments", "organic"]
ACOUSTIC_FALSE = ["electronic", "synth", "synthy", "produced"]

# Genres people ask for that are NOT in the catalog. Recognising them lets the
# guardrail tell the user "we don't have that" instead of silently ignoring it.
UNSUPPORTED_GENRES = [
    "k pop", "kpop", "j pop", "jpop", "opera", "gospel", "punk", "grunge", "disco",
    "funk", "bluegrass", "emo", "drill", "afrobeats", "afrobeat", "christmas", "anime",
    "bollywood", "polka", "drum and bass",
]

# What "not X" should mean when X is a mood/energy word rather than a genre.
NEGATED_HINTS = {
    "sad": {"valence": 0.75}, "depressing": {"valence": 0.75}, "melancholy": {"valence": 0.75},
    "intense": {"energy": 0.45}, "loud": {"energy": 0.45}, "aggressive": {"energy": 0.45},
    "fast": {"energy": 0.45}, "hype": {"energy": 0.45},
    "slow": {"energy": 0.7}, "boring": {"energy": 0.7}, "sleepy": {"energy": 0.7},
    "happy": {"valence": 0.35}, "upbeat": {"energy": 0.4},
}

# A second situation doc only fills gaps if it scored close to the best one.
SECONDARY_DOC_RATIO = 0.8

NEGATORS = r"(?:no|not|without|anything but|except|avoid|skip|hate|nothing)"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}


@dataclass
class Plan:
    profile: Dict
    sources: Dict[str, str] = field(default_factory=dict)   # field -> where it came from
    notes: List[str] = field(default_factory=list)
    seed_song_id: Optional[int] = None
    planner: str = "rules"


def _has(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def split_negations(text: str) -> Tuple[str, List[str]]:
    """
    Pull out negated phrases like "no country" or "not too sad".

    Returns (text_without_negated_words, negated_phrases). The cleaned text is
    what gets sent to the retriever, so "not sad" does not retrieve the
    heartbreak document.
    """
    norm = normalize(text)
    negated: List[str] = []
    # Capture one word, plus a second one only for multi-word names like
    # "hip hop". The lookahead stops it from swallowing the next negation
    # ("no country nothing sad") or filler words ("no rap for the gym").
    second = rf"(?:\s+(?!(?:{NEGATORS}|and|but|or|for|to|at|in|on|with|while|the|a|music|songs|please)\b)([a-z&]+))?"
    pattern = re.compile(rf"\b{NEGATORS}\s+(?:any\s+|too\s+|more\s+)?([a-z&]+){second}")

    def repl(m: re.Match) -> str:
        negated.append(" ".join(g for g in m.groups() if g))
        return " "

    cleaned = pattern.sub(repl, norm)
    return re.sub(r"\s+", " ", cleaned).strip(), negated


def _negated_genres(negated: List[str], genre_docs: List) -> List[str]:
    """Map negated phrases to catalog genres using the genre docs' keywords."""
    out = []
    for phrase in negated:
        for doc in genre_docs:
            for kw in doc.keywords:
                if phrase == kw or phrase.split()[0] == kw:
                    out.append(str(doc.hints["genre"]))
    return sorted(set(out))


class RulePlanner:
    name = "rules"

    def __init__(self, genre_docs: List):
        # genre docs are needed to understand "no rap" -> exclude hip hop
        self.genre_docs = genre_docs

    def plan(self, request: str, retrieved: List[RetrievedDoc]) -> Plan:
        text = normalize(request)
        positive_text, negated = split_negations(request)
        plan = Plan(profile={})
        prof, src = plan.profile, plan.sources

        def put(key, value, source, override=True):
            if override or key not in prof:
                prof[key] = value
                src[key] = source

        # 1. situation docs, best match first; later ones only fill gaps, and
        #    only if they matched about as strongly as the best one
        situations = [h for h in retrieved if h.doc.source == "situations"]
        for hit in situations:
            if hit.score < situations[0].score * SECONDARY_DOC_RATIO:
                continue
            for key, value in hit.doc.hints.items():
                put(key, value, f"situation: {hit.doc.title}", override=False)

        # 2. a song the user named ("like Library Rain")
        songs = [h for h in retrieved if h.doc.source == "catalog"]
        if songs:
            seed = songs[0].doc
            plan.seed_song_id = seed.song["id"]
            for key, value in seed.hints.items():
                put(key, value, f"song: {seed.title}")
            plan.notes.append(f"Using '{seed.title}' as a reference song and leaving it out of the results.")

        # 3. genre docs (explicit genre names / aliases)
        excluded = _negated_genres(negated, self.genre_docs)
        genre_hits = [h for h in retrieved if h.doc.source == "genres"
                      and h.doc.hints["genre"] not in excluded]
        if genre_hits:
            put("genre", genre_hits[0].doc.hints["genre"], f"genre guide: {genre_hits[0].doc.title}")
            if len(genre_hits) > 1:
                plan.notes.append(
                    f"You mentioned more than one genre; the scorer weights one, so I used "
                    f"{genre_hits[0].doc.title} and let mood/energy pull in the rest."
                )
        else:
            for g in UNSUPPORTED_GENRES:
                if _has(positive_text, g):
                    put("genre", g, "request text")
                    break
        if excluded:
            prof["exclude_genres"] = excluded
            src["exclude_genres"] = "request text"
            if prof.get("genre") in excluded:
                prof.pop("genre")

        # 4. explicit words in the request beat everything above
        for mood, words in MOOD_WORDS.items():
            if any(_has(positive_text, w) for w in words):
                put("mood", mood, "request text")
                break
        for value, words in ENERGY_WORDS.items():
            if any(_has(positive_text, w) for w in words):
                put("energy", value, "request text")
                break
        if any(_has(positive_text, w) for w in ACOUSTIC_TRUE):
            put("likes_acoustic", True, "request text")
        elif any(_has(positive_text, w) for w in ACOUSTIC_FALSE) or any(
            n.startswith("acoustic") for n in negated
        ):
            put("likes_acoustic", False, "request text")

        # 5. negated mood/energy words ("not too intense", "nothing sad")
        for phrase in negated:
            for key, value in NEGATED_HINTS.get(phrase.split()[0], {}).items():
                put(key, value, f"request text (not {phrase.split()[0]})")

        k = _extract_k(text)
        if k is not None:
            put("k", k, "request text")

        return plan


def _extract_k(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")\b", text)
    if not m:
        m = re.search(r"\b(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")\s+(?:songs|tracks|picks|recommendations|recs)\b", text)
    if not m:
        return None
    raw = m.group(1)
    return NUMBER_WORDS.get(raw, None) or int(raw)
