"""
VibeFinder command line app.

    python -m src.vibefinder "chill lofi for studying"
    python -m src.vibefinder "gym music, no metal" --trace
    python -m src.vibefinder --rules "..."    # skip Gemini even if a key is set
    python -m src.vibefinder                  # interactive mode
"""

import argparse
import json

from src.agent import AgentResult, VibeFinderAgent, configure_logging
from src.formatting import table


def render(result: AgentResult, show_trace: bool = False) -> str:
    out = [f'\nRequest: "{result.request}"']
    if not result.ok:
        out.append(f"No recommendations. {result.message}")
        for w in result.warnings:
            out.append(f"  ! {w}")
        if show_trace:
            out.append(render_trace(result))
        return "\n".join(out)

    prof = {k: v for k, v in result.profile.items() if k != "k"}
    out.append(f"Planner: {result.planner} | Strategy: {result.strategy} | "
               f"Confidence: {result.confidence:.2f} ({result.confidence_label})")
    out.append(f"Profile used: {prof}")
    rows = [
        (r.rank, f"{r.title} - {r.artist}", f"{r.genre}/{r.mood}", f"{r.score:.2f}",
         f"{r.fit:.0%}", r.reasons)
        for r in result.recommendations
    ]
    out.append(table(["#", "Song", "Genre/Mood", "Score", "Fit", "Why"], rows, max_width=48))
    if result.grounding:
        out.append("Based on these notes:")
        for g in result.grounding:
            out.append(f"  - {g}")
    if result.warnings:
        out.append("Warnings:")
        for w in result.warnings:
            out.append(f"  ! {w}")
    if show_trace:
        out.append(render_trace(result))
    return "\n".join(out)


def render_trace(result: AgentResult) -> str:
    lines = ["Agent trace:"]
    for i, step in enumerate(result.trace, start=1):
        lines.append(f"  {i}. {step['step']}: {json.dumps(step['detail'], default=str)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask VibeFinder for music in plain English.")
    parser.add_argument("request", nargs="*", help="what you want to listen to")
    parser.add_argument("--rules", action="store_true", help="don't use Gemini, rules only")
    parser.add_argument("--trace", action="store_true", help="print the agent's step-by-step trace")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show info logs in the console")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    agent = VibeFinderAgent(use_llm=not args.rules)

    def answer(text: str) -> None:
        result = agent.run(text)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(render(result, args.trace))

    if args.request:
        answer(" ".join(args.request))
        return

    mode = "Gemini + rules fallback" if agent.llm_planner else "rules"
    print(f"VibeFinder ({mode}). Describe what you want to hear, or 'quit' to exit.")
    while True:
        try:
            text = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            break
        answer(text)


if __name__ == "__main__":
    main()
