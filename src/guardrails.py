"""
Guardrails: checks that run before and after the AI-ish parts of the system.

Input guardrail  -> is this a usable music request at all?
Profile guardrail -> did the planner (rules or LLM) produce a sane profile
                     that the scorer can actually use with this catalog?

Both return plain data plus a list of human readable warnings, so the CLI
can show the user exactly what was changed and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

MAX_REQUEST_CHARS = 300
MIN_K, MAX_K, DEFAULT_K = 1, 10, 5

# Phrases that look like someone trying to steer an LLM instead of asking
# for music. We don't block the request, we just strip the phrase and warn.
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions?",
    r"you are now",
    r"system prompt",
    r"disregard (the )?rules",
]

ALLOWED_KEYS = {"genre", "mood", "energy", "likes_acoustic", "tempo_bpm", "valence", "k",
                "exclude_genres"}


@dataclass
class InputCheck:
    ok: bool
    text: str
    reason: str = ""
    warnings: List[str] = field(default_factory=list)


def check_request(raw: Optional[str]) -> InputCheck:
    """Reject empty / junk input, trim very long input, strip injection phrases."""
    if raw is None or not raw.strip():
        return InputCheck(False, "", "Empty request. Try something like 'chill music for studying'.")

    text = " ".join(raw.split())
    warnings: List[str] = []

    if len(text) > MAX_REQUEST_CHARS:
        text = text[:MAX_REQUEST_CHARS]
        warnings.append(f"Request was longer than {MAX_REQUEST_CHARS} characters and was cut off.")

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            text = re.sub(pattern, " ", text, flags=re.I)
            msg = "Removed text that looked like instructions to the AI rather than a music request."
            if msg not in warnings:
                warnings.append(msg)
    text = " ".join(text.split())

    if not re.search(r"[A-Za-z]{2,}", text):
        return InputCheck(False, text, "That doesn't look like a request I can read. Use a few words about the music you want.", warnings)

    return InputCheck(True, text, "", warnings)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def validate_profile(
    profile: Dict,
    known_genres: Iterable[str],
    known_moods: Iterable[str],
) -> Tuple[Dict, List[str]]:
    """
    Clean a preference profile so the scorer never sees bad values.

    - unknown keys are dropped
    - genre / mood not in the catalog are dropped (with a warning), because
      a genre nobody can match just adds 0 to every song and hides the issue
    - numbers are coerced and clamped into range
    - k is forced into 1..10
    """
    known_genres = {g.lower() for g in known_genres}
    known_moods = {m.lower() for m in known_moods}
    clean: Dict = {}
    warnings: List[str] = []

    for key in profile:
        if key not in ALLOWED_KEYS:
            warnings.append(f"Ignored unexpected field '{key}'.")

    genre = profile.get("genre")
    if genre:
        genre = str(genre).strip().lower()
        if genre in known_genres:
            clean["genre"] = genre
        else:
            warnings.append(
                f"The catalog has no '{genre}' songs, so genre was left out and the "
                f"other preferences decide the ranking."
            )

    mood = profile.get("mood")
    if mood:
        mood = str(mood).strip().lower()
        if mood in known_moods:
            clean["mood"] = mood
        else:
            warnings.append(f"No songs are tagged with the mood '{mood}', so mood was left out.")

    for key, low, high in (("energy", 0.0, 1.0), ("valence", 0.0, 1.0), ("tempo_bpm", 40.0, 220.0)):
        if profile.get(key) is None:
            continue
        try:
            value = float(profile[key])
        except (TypeError, ValueError):
            warnings.append(f"'{key}' was not a number and was ignored.")
            continue
        if not low <= value <= high:
            margin = (high - low) * 0.1
            if value < low - margin or value > high + margin:
                # Way out of range usually means the planner used a different
                # scale (e.g. energy 7 out of 10). Guessing is worse than skipping.
                warnings.append(f"'{key}' was {value}, far outside {low}-{high}; ignored it.")
                continue
            warnings.append(f"'{key}' was {value}, slightly outside {low}-{high}; clamped.")
        clean[key] = round(_clamp(value, low, high), 3)

    if profile.get("likes_acoustic") is not None:
        val = profile["likes_acoustic"]
        if isinstance(val, str):
            val = val.strip().lower() in ("true", "yes", "1")
        clean["likes_acoustic"] = bool(val)

    excludes = [str(g).lower() for g in profile.get("exclude_genres", []) or []]
    excludes = [g for g in excludes if g in known_genres]
    if excludes:
        clean["exclude_genres"] = sorted(set(excludes))
        if clean.get("genre") in clean["exclude_genres"]:
            warnings.append(f"You both asked for and excluded '{clean['genre']}'; the exclusion wins.")
            clean.pop("genre")

    k = profile.get("k", DEFAULT_K)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = DEFAULT_K
    if not MIN_K <= k <= MAX_K:
        warnings.append(f"Asked for {k} songs; using {int(_clamp(k, MIN_K, MAX_K))} (limit is {MIN_K}-{MAX_K}).")
    clean["k"] = int(_clamp(k, MIN_K, MAX_K))

    return clean, warnings


def has_preferences(profile: Dict) -> bool:
    """True if the profile has at least one thing the scorer can use."""
    return any(profile.get(k) is not None for k in ("genre", "mood", "energy", "likes_acoustic", "tempo_bpm", "valence"))
