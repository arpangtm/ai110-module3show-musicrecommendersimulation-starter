"""
Command line runner for the base Music Recommender Simulation.

Runs a handful of hand-written listener profiles through the scoring
recommender and prints a table for each one.

    python -m src.main                 # all profiles
    python -m src.main --profile gym   # just one
    python -m src.main --no-diversity  # turn off the artist penalty
"""

import argparse
from pathlib import Path

from src.formatting import table
from src.recommender import load_songs, recommend_songs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

PROFILES = {
    "pop": {"genre": "pop", "mood": "happy", "energy": 0.8},
    "gym": {"genre": "edm", "mood": "euphoric", "energy": 0.95, "likes_acoustic": False},
    "study": {"genre": "lofi", "mood": "focused", "energy": 0.35, "likes_acoustic": True},
    "acoustic_sad": {"genre": "folk", "mood": "melancholy", "energy": 0.3, "likes_acoustic": True},
    "rap": {"genre": "hip hop", "mood": "confident", "energy": 0.75, "tempo_bpm": 95},
}


def show(name: str, prefs: dict, songs: list, k: int, diversity: bool) -> None:
    recs = recommend_songs(prefs, songs, k=k, diversity=diversity)
    print(f"\nProfile: {name}  {prefs}")
    rows = [
        (i, f"{s['title']} - {s['artist']}", f"{s['genre']}/{s['mood']}", f"{score:.2f}", why)
        for i, (s, score, why) in enumerate(recs, start=1)
    ]
    print(table(["#", "Song", "Genre/Mood", "Score", "Why"], rows, max_width=60))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the base music recommender.")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="run a single profile")
    parser.add_argument("-k", type=int, default=5, help="number of songs to return")
    parser.add_argument("--no-diversity", action="store_true", help="disable artist penalty")
    args = parser.parse_args()

    songs = load_songs(str(DATA_PATH))
    names = [args.profile] if args.profile else list(PROFILES)
    for name in names:
        show(name, PROFILES[name], songs, args.k, not args.no_diversity)


if __name__ == "__main__":
    main()
