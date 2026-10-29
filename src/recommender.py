"""
Core scoring and ranking logic for the music recommender.

This is the "base project" (Module 3). Everything in here is deterministic:
given the same catalog and the same user preferences, you always get the
same ranked list back, along with the reasons each song scored the way it did.

The AI layer in src/agent.py sits on top of this module. It turns a plain
English request into a preference dict, then calls recommend_songs().
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default weights. Kept in one place so experiments (and the agent's
# self-critique step) can swap them without touching the scoring code.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre": 2.0,
    "mood": 1.5,
    "energy": 2.0,
    "acoustic": 1.0,
    "tempo": 0.5,
    "valence": 0.5,
}

# Penalty applied when the same artist would show up more than once
# in a single list of recommendations.
ARTIST_REPEAT_PENALTY = 1.0

NUMERIC_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")


@dataclass
class Song:
    """A song and its audio/descriptive attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """A listener's taste preferences (OOP version used by the tests)."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    def to_prefs(self) -> Dict:
        return {
            "genre": self.favorite_genre,
            "mood": self.favorite_mood,
            "energy": self.target_energy,
            "likes_acoustic": self.likes_acoustic,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file into a list of dicts.

    Numeric columns are converted to floats (id to int). Rows with missing or
    non-numeric values are skipped and logged instead of crashing the program.
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            try:
                song = dict(row)
                song["id"] = int(row["id"])
                for field in NUMERIC_FIELDS:
                    song[field] = float(row[field])
                song["genre"] = row["genre"].strip().lower()
                song["mood"] = row["mood"].strip().lower()
                songs.append(song)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping bad row %d in %s: %s", line_no, csv_path, exc)
    logger.info("Loaded %d songs from %s", len(songs), csv_path)
    return songs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _genre_points(wanted: str, actual: str, weight: float) -> Tuple[float, Optional[str]]:
    if not wanted:
        return 0.0, None
    wanted, actual = wanted.lower(), actual.lower()
    if wanted == actual:
        return weight, f"genre matches {actual} (+{weight:.2f})"
    # "indie pop" should still count a little for someone who asked for "pop"
    if wanted in actual.split() or actual in wanted.split():
        pts = weight / 2
        return pts, f"related genre {actual} (+{pts:.2f})"
    return 0.0, None


def max_possible_score(user_prefs: Dict, weights: Optional[Dict[str, float]] = None) -> float:
    """Best score a perfect song could get for these preferences."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = 0.0
    if user_prefs.get("genre"):
        total += w["genre"]
    if user_prefs.get("mood"):
        total += w["mood"]
    if user_prefs.get("energy") is not None:
        total += w["energy"]
    if user_prefs.get("likes_acoustic") is not None:
        total += w["acoustic"]
    if user_prefs.get("tempo_bpm") is not None:
        total += w["tempo"]
    if user_prefs.get("valence") is not None:
        total += w["valence"]
    return total


def score_song(
    user_prefs: Dict,
    song: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[str]]:
    """
    Score one song against a user's preferences.

    user_prefs keys (all optional except you need at least one):
        genre (str), mood (str), energy (0-1), likes_acoustic (bool),
        tempo_bpm (number), valence (0-1)

    Returns (score, reasons) where reasons is a list of short strings that
    say exactly which rule added points.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    score = 0.0
    reasons: List[str] = []

    pts, why = _genre_points(user_prefs.get("genre", ""), song["genre"], w["genre"])
    if why:
        score += pts
        reasons.append(why)

    mood = (user_prefs.get("mood") or "").lower()
    if mood and mood == song["mood"]:
        score += w["mood"]
        reasons.append(f"mood is {mood} (+{w['mood']:.2f})")

    target_energy = user_prefs.get("energy")
    if target_energy is not None:
        closeness = 1.0 - abs(float(target_energy) - song["energy"])
        pts = w["energy"] * closeness
        score += pts
        reasons.append(
            f"energy {song['energy']:.2f} vs target {float(target_energy):.2f} (+{pts:.2f})"
        )

    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        fit = song["acousticness"] if likes_acoustic else 1.0 - song["acousticness"]
        pts = w["acoustic"] * fit
        score += pts
        label = "acoustic" if likes_acoustic else "produced/electronic"
        reasons.append(f"{label} sound fit {fit:.2f} (+{pts:.2f})")

    target_tempo = user_prefs.get("tempo_bpm")
    if target_tempo is not None:
        # Within ~60 bpm counts for something, exact match gets full points.
        closeness = max(0.0, 1.0 - abs(float(target_tempo) - song["tempo_bpm"]) / 60.0)
        pts = w["tempo"] * closeness
        score += pts
        reasons.append(f"tempo {song['tempo_bpm']:.0f} bpm (+{pts:.2f})")

    target_valence = user_prefs.get("valence")
    if target_valence is not None:
        closeness = 1.0 - abs(float(target_valence) - song["valence"])
        pts = w["valence"] * closeness
        score += pts
        reasons.append(f"positivity {song['valence']:.2f} (+{pts:.2f})")

    return round(score, 4), reasons


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    diversity: bool = True,
) -> List[Tuple[Dict, float, str]]:
    """
    Rank the catalog and return the top k as (song, score, explanation).

    If diversity is on, the list is built greedily and any song by an artist
    already in the list loses ARTIST_REPEAT_PENALTY points. This stops one
    artist from taking over the whole top 5.
    """
    if k <= 0 or not songs:
        return []

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights)
        scored.append((song, score, reasons))

    # Sort by score, then by title so ties are always broken the same way.
    scored.sort(key=lambda item: (-item[1], item[0]["title"]))

    if not diversity:
        return [(s, sc, "; ".join(r)) for s, sc, r in scored[:k]]

    picked: List[Tuple[Dict, float, str]] = []
    seen_artists = set()
    remaining = list(scored)
    while remaining and len(picked) < k:
        best_idx, best_adj, best_reasons = 0, None, None
        for idx, (song, score, reasons) in enumerate(remaining):
            adj = score
            r = list(reasons)
            if song["artist"] in seen_artists:
                adj -= ARTIST_REPEAT_PENALTY
                r.append(f"artist already in list (-{ARTIST_REPEAT_PENALTY:.2f})")
            if best_adj is None or adj > best_adj:
                best_idx, best_adj, best_reasons = idx, adj, r
        song = remaining.pop(best_idx)[0]
        seen_artists.add(song["artist"])
        picked.append((song, round(best_adj, 4), "; ".join(best_reasons)))
    return picked


class Recommender:
    """OOP wrapper around the functional recommender."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _as_dicts(self) -> List[Dict]:
        return [asdict(s) for s in self.songs]

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = user.to_prefs()
        ranked = recommend_songs(prefs, self._as_dicts(), k=k)
        by_id = {s.id: s for s in self.songs}
        return [by_id[song["id"]] for song, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(user.to_prefs(), asdict(song))
        if not reasons:
            return f"{song.title} did not match any of your preferences (score 0)."
        return f"{song.title} scored {score:.2f} because: " + "; ".join(reasons)
