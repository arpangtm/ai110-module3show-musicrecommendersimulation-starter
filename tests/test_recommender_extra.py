"""Tests for the base scoring recommender (data loading, scoring, ranking)."""

from pathlib import Path

from src.recommender import load_songs, recommend_songs, score_song, max_possible_score

DATA = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def test_catalog_loads_twenty_songs_with_numeric_fields():
    songs = load_songs(str(DATA))
    assert len(songs) == 20
    for s in songs:
        assert isinstance(s["energy"], float) and 0 <= s["energy"] <= 1
        assert isinstance(s["id"], int)


def test_bad_csv_rows_are_skipped_not_fatal(tmp_path):
    bad = tmp_path / "songs.csv"
    bad.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,Good,A,pop,happy,0.5,100,0.5,0.5,0.5\n"
        "2,Broken,B,pop,happy,not_a_number,100,0.5,0.5,0.5\n"
    )
    songs = load_songs(str(bad))
    assert [s["title"] for s in songs] == ["Good"]


def test_score_is_numeric_and_never_exceeds_max_for_every_song():
    songs = load_songs(str(DATA))
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    best = max_possible_score(prefs)
    for s in songs:
        score, reasons = score_song(prefs, s)
        assert isinstance(score, float)
        assert 0 <= score <= best
        assert isinstance(reasons, list)


def test_energy_preference_actually_changes_ranking():
    songs = load_songs(str(DATA))
    loud = recommend_songs({"energy": 0.95}, songs, k=3)
    quiet = recommend_songs({"energy": 0.15}, songs, k=3)
    assert all(s["energy"] > 0.85 for s, _, _ in loud)
    assert all(s["energy"] < 0.35 for s, _, _ in quiet)


def test_results_are_sorted_and_explained():
    songs = load_songs(str(DATA))
    recs = recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.4}, songs, k=5)
    scores = [score for _, score, _ in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(why for _, _, why in recs)
    assert recs[0][0]["genre"] == "lofi"


def test_artist_penalty_spreads_out_artists():
    songs = load_songs(str(DATA))
    prefs = {"genre": "lofi", "energy": 0.4}
    with_div = recommend_songs(prefs, songs, k=3, diversity=True)
    without = recommend_songs(prefs, songs, k=3, diversity=False)
    artists_with = [s["artist"] for s, _, _ in with_div]
    artists_without = [s["artist"] for s, _, _ in without]
    assert artists_without.count("LoRoom") == 2
    assert len(set(artists_with)) >= len(set(artists_without))
    assert any("artist already in list" in why for _, _, why in with_div) or len(set(artists_with)) == 3


def test_k_zero_and_empty_catalog_return_empty():
    assert recommend_songs({"genre": "pop"}, [], k=5) == []
    assert recommend_songs({"genre": "pop"}, load_songs(str(DATA)), k=0) == []
