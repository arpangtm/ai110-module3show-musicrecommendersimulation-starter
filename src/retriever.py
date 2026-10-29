"""
Retrieval step of the RAG pipeline.

Three sources get searched for every request:

  1. situations  (knowledge/situations.md)  what people tend to want for
                 an activity or feeling: studying, gym, rainy day, ...
  2. genres      (knowledge/genres.md)      every genre in the catalog plus
                 the other names people use for it (rap -> hip hop)
  3. catalog     (data/songs.csv)           song titles, so a request like
                 "more like Library Rain" can be grounded in a real song

Each knowledge section has a `hints:` line. Those hints are what the planner
turns into a preference profile, and the section text is quoted back in the
explanation, so the retrieved context directly changes the output.

The scoring is intentionally simple and inspectable: keyword phrase hits plus
a small bonus for word overlap with the section body. No embeddings, so it
runs offline and gives the same answer every time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "in", "on", "me", "my", "i",
    "some", "something", "songs", "song", "music", "play", "want", "need",
    "give", "with", "that", "is", "it", "be", "like", "can", "you", "please",
    "while", "im", "am", "are", "about", "just", "really", "get", "good",
    "while", "at", "or", "this", "what", "your", "any", "track", "tracks",
}

KEYWORD_HIT = 2.0          # each matched keyword phrase
MULTIWORD_BONUS = 0.5      # extra per additional word in a matched phrase
BODY_OVERLAP = 0.25        # per shared content word with the section body
BODY_OVERLAP_CAP = 1.0
MIN_SCORE = 2.0            # below this a document is not considered relevant
TITLE_HIT = 5.0            # exact song title mentioned in the request


@dataclass
class Document:
    doc_id: str
    source: str            # "situations" | "genres" | "catalog"
    title: str
    keywords: List[str]
    hints: Dict[str, object]
    text: str
    song: Optional[Dict] = None   # set for catalog documents


@dataclass
class RetrievedDoc:
    doc: Document
    score: float
    matched: List[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """Lowercase and strip punctuation, keeping '&' so 'r&b' survives."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9& ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set:
    return {_stem(w) for w in normalize(text).split() if w not in STOPWORDS and len(w) > 2}


def parse_hints(line: str) -> Dict[str, object]:
    """Turn 'genre=lofi; energy=0.35; likes_acoustic=true' into a dict."""
    hints: Dict[str, object] = {}
    for part in line.split(";"):
        if "=" not in part:
            continue
        key, value = (x.strip() for x in part.split("=", 1))
        if key in ("energy", "valence", "tempo_bpm"):
            hints[key] = float(value)
        elif key == "likes_acoustic":
            hints[key] = value.lower() == "true"
        else:
            hints[key] = value.lower()
    return hints


def load_markdown_docs(path: Path, source: str) -> List[Document]:
    """Split a knowledge file on '## ' headings, one Document per section."""
    docs: List[Document] = []
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
    for i, section in enumerate(sections):
        lines = section.strip().splitlines()
        title = lines[0].strip()
        keywords: List[str] = []
        hints: Dict[str, object] = {}
        body: List[str] = []
        for line in lines[1:]:
            if line.lower().startswith("keywords:"):
                keywords = [normalize(k) for k in line.split(":", 1)[1].split(",") if k.strip()]
            elif line.lower().startswith("hints:"):
                hints = parse_hints(line.split(":", 1)[1])
            elif line.strip():
                body.append(line.strip())
        docs.append(Document(f"{source}:{i}", source, title, keywords, hints, " ".join(body)))
    logger.info("Loaded %d documents from %s", len(docs), path.name)
    return docs


def catalog_docs(songs: List[Dict]) -> List[Document]:
    docs = []
    for s in songs:
        hints = {
            "genre": s["genre"],
            "mood": s["mood"],
            "energy": s["energy"],
            "likes_acoustic": s["acousticness"] >= 0.5,
            "tempo_bpm": s["tempo_bpm"],
        }
        text = f"{s['title']} by {s['artist']}, a {s['mood']} {s['genre']} track."
        docs.append(
            Document(f"catalog:{s['id']}", "catalog", s["title"], [normalize(s["title"])], hints, text, s)
        )
    return docs


class Retriever:
    """Keyword + overlap retriever over the knowledge files and the catalog."""

    def __init__(self, knowledge_dir: Path, songs: List[Dict]):
        self.docs: List[Document] = []
        self.docs += load_markdown_docs(knowledge_dir / "situations.md", "situations")
        self.docs += load_markdown_docs(knowledge_dir / "genres.md", "genres")
        self.docs += catalog_docs(songs)

    def score(self, query: str, doc: Document) -> RetrievedDoc:
        q = f" {normalize(query)} "
        score, matched = 0.0, []
        for kw in doc.keywords:
            if kw and f" {kw} " in q:
                if doc.source == "catalog":
                    score += TITLE_HIT
                else:
                    score += KEYWORD_HIT + MULTIWORD_BONUS * (len(kw.split()) - 1)
                matched.append(kw)
        if matched and doc.source != "catalog":
            overlap = content_words(query) & content_words(doc.title + " " + doc.text)
            score += min(BODY_OVERLAP_CAP, BODY_OVERLAP * len(overlap))
        return RetrievedDoc(doc, round(score, 3), matched)

    def search(self, query: str, per_source: int = 2) -> List[RetrievedDoc]:
        """Return relevant docs, best first, at most per_source from each source."""
        # Song titles first. If the user named a song, take its title out of the
        # query so words inside it ("Rain" in "Library Rain") don't also pull in
        # unrelated situation notes.
        title_hits = [self.score(query, d) for d in self.docs if d.source == "catalog"]
        title_hits = [h for h in title_hits if h.score >= MIN_SCORE]
        rest = f" {normalize(query)} "
        for h in title_hits:
            rest = rest.replace(f" {h.doc.keywords[0]} ", " ")
        hits = title_hits + [self.score(rest, d) for d in self.docs if d.source != "catalog"]
        hits = [h for h in hits if h.score >= MIN_SCORE]
        hits.sort(key=lambda h: (-h.score, h.doc.doc_id))
        out, counts = [], {}
        for h in hits:
            if counts.get(h.doc.source, 0) < per_source:
                out.append(h)
                counts[h.doc.source] = counts.get(h.doc.source, 0) + 1
        logger.info(
            "Retrieved for %r: %s", query, [(h.doc.title, h.score) for h in out]
        )
        return out
