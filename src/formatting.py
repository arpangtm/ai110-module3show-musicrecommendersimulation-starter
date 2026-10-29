"""Plain-text table output so results are easy to read in a terminal."""

from typing import List, Sequence


def table(headers: Sequence[str], rows: List[Sequence], max_width: int = 70) -> str:
    """Render rows as a simple ASCII table. Long cells are wrapped."""
    def wrap(text: str, width: int) -> List[str]:
        words, lines, cur = str(text).split(), [], ""
        for word in words:
            if cur and len(cur) + 1 + len(word) > width:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        lines.append(cur)
        return lines or [""]

    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = min(max(widths[i], len(c)), max_width)

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |", sep]
    for r in str_rows:
        wrapped = [wrap(c, widths[i]) for i, c in enumerate(r)]
        height = max(len(x) for x in wrapped)
        for line in range(height):
            cells = [
                (wrapped[i][line] if line < len(wrapped[i]) else "").ljust(widths[i])
                for i in range(len(r))
            ]
            out.append("| " + " | ".join(cells) + " |")
        out.append(sep)
    return "\n".join(out)
