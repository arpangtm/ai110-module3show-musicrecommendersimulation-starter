# AI Interactions Log

This file covers the stretch features I attempted for Project 4: the agentic workflow enhancement (with saved reasoning traces), the evaluation harness, and multi-source retrieval. It also records how I used an AI agent to build the project.

---

## Agentic Workflow Enhancement

### What the agent in my system does

VibeFinder's agent (`src/agent.py`) runs a decision chain on every request instead of a single function call:

1. **check_input** - is the request usable? strip injection text, flag crisis language
2. **retrieve** - search three sources (situation notes, genre guide, song titles)
3. **plan** - turn the request plus retrieved hints into a preference profile (Gemini if configured, rules otherwise, rules as a fallback if Gemini fails)
4. **validate_profile** - drop anything the catalog can't use, clamp or drop bad numbers
5. **recommend** - run the base scorer as a tool
6. **critique** - grade its own list: confidence, is the requested mood in the top 3, is the energy close
7. **revise** - if there are issues, switch weighting strategy (mood_first or energy_first) and call the scorer again, up to 2 more times
8. **decide** - compare all attempts and keep the one with the fewest issues, then the highest confidence
9. **done** - write the whole trace to `logs/agent_trace.jsonl`

The important part for me was step 8. My first version just kept the last attempt, which was sometimes worse than the first one. Comparing attempts made the loop actually useful.

### Saved reasoning traces

Full traces are committed in two places:

- [`assets/agent_traces.txt`](assets/agent_traces.txt) - readable output from `python -m src.vibefinder --trace` for three requests
- [`assets/agent_trace_sample.jsonl`](assets/agent_trace_sample.jsonl) - the raw JSON log the agent writes, one line per request, for all the sample runs

Two of them embedded here:

**Trace 1: a conflicting request, where the agent revises twice and then admits it can't fully satisfy it**

```
$ python -m src.vibefinder --trace "slow happy songs"

Agent trace:
  1. check_input: {"ok": true, "text": "slow happy songs", "reason": ""}
  2. retrieve: {"query": "slow happy songs", "negated": [], "hits": []}
  3. plan: "Rule planner built {'mood': 'happy', 'energy': 0.22} from {'mood': 'request text', 'energy': 'request text'}"
  4. validate_profile: {"profile": {"mood": "happy", "energy": 0.22, "k": 5}, "warnings": []}
  5. recommend: {"strategy": "balanced", "top": ["Rooftop Lights", "Sunrise City", "Moonlit Sonata Study", "Spacewalk Thoughts", "Pine Smoke"], "confidence": 0.636, "issues": ["energy_mismatch"]}
  6. revise: {"strategy": "energy_first", "top": ["Moonlit Sonata Study", "Spacewalk Thoughts", "Pine Smoke", "Rooftop Lights", "Library Rain"], "confidence": 0.663, "issues": ["mood_missing_in_top3"]}
  7. revise: {"strategy": "mood_first", "top": ["Rooftop Lights", "Sunrise City", "Moonlit Sonata Study", "Spacewalk Thoughts", "Pine Smoke"], "confidence": 0.645, "issues": ["energy_mismatch"]}
  8. decide: {"chosen_strategy": "energy_first", "compared": {"balanced": {"confidence": 0.636, "issues": ["energy_mismatch"]}, "energy_first": {"confidence": 0.663, "issues": ["mood_missing_in_top3"]}, "mood_first": {"confidence": 0.645, "issues": ["energy_mismatch"]}}}
  9. done: {"ok": true, "seconds": 0.0014}
```

What's going on: the balanced ranking put happy songs on top, but they're all high energy, so the critic flagged an energy mismatch. Energy-first fixed the energy but lost the mood. Mood-first went back to the energy problem. Every option had one issue, so the agent picked the one with the best confidence and the user gets a warning that the catalog can't do both.

**Trace 2: two retrieved notes combining, plus an exclusion**

```
$ python -m src.vibefinder --trace "rainy day study session, no jazz"

Agent trace:
  1. check_input: {"ok": true, "text": "rainy day study session, no jazz", "reason": ""}
  2. retrieve: {"query": "rainy day study session", "negated": ["jazz"], "hits": [{"source": "situations", "title": "Rainy day", "score": 2.5, "matched": ["rainy"]}, {"source": "situations", "title": "Studying and deep focus", "score": 2.25, "matched": ["study"]}]}
  3. plan: "Rule planner built {'mood': 'melancholy', 'energy': 0.35, 'likes_acoustic': True, 'genre': 'lofi', 'exclude_genres': ['jazz']} from {'mood': 'situation: Rainy day', 'energy': 'situation: Rainy day', 'likes_acoustic': 'situation: Rainy day', 'genre': 'situation: Studying and deep focus', 'exclude_genres': 'request text'}"
  4. validate_profile: {"profile": {"genre": "lofi", "mood": "melancholy", "energy": 0.35, "likes_acoustic": true, "exclude_genres": ["jazz"], "k": 5}, "warnings": []}
  5. recommend: {"strategy": "balanced", "top": ["Library Rain", "Focus Flow", "Pine Smoke", "Rainy Day Blues", "Midnight Coding"], "confidence": 0.711, "issues": []}
  6. decide: {"chosen_strategy": "balanced", "compared": {"balanced": {"confidence": 0.711, "issues": []}}}
  7. done: {"ok": true, "seconds": 0.0014}
```

The plan step shows where each field came from. Mood, energy and acoustic came from the rainy day note (the stronger match), and genre came from the studying note, which only filled the gap. "Jazz" was pulled out before retrieval so it became an exclusion instead of matching the jazz genre note.

---

## Test Harness / Evaluation Script

`python -m eval.run_eval` runs 17 predefined requests, checks each one against a written expectation, runs each request a second time to make sure the answer doesn't change, and prints a summary. It exits with code 1 if anything fails so it could run in CI.

```
17/17 checks passed | 17/17 gave identical results on a second run | average confidence on answered requests: 0.71
```

Full table: [`assets/eval_report.md`](assets/eval_report.md). Raw console output: [`assets/eval_output.txt`](assets/eval_output.txt).

---

## RAG Enhancement: multi-source retrieval

Retrieval searches three different sources, each doing a different job:

| Source | File | What it adds |
|---|---|---|
| Situation notes | `knowledge/situations.md` | Turns activities and feelings into numbers ("rainy day" means melancholy, low energy, acoustic) |
| Genre guide | `knowledge/genres.md` | Maps the way people actually say genres onto catalog genres ("rap" means hip hop, "lo-fi" means lofi) |
| Song titles | `data/songs.csv` | Lets someone ask for "more like Library Rain" and uses that song's features as the profile |

Before/after is documented in the README ("Retrieval on vs off"). The short version: with retrieval off, 4 of 5 everyday requests ("music for my workout", "help me fall asleep") get refused because they contain no genre or mood word. With it on, all 5 get answered, and "songs for a romantic dinner" goes from 0.50 to 0.84 confidence.

One thing I had to fix: "more like Library Rain" was also retrieving the rainy day note because of the word "Rain" in the title. Now song titles are matched first and removed from the query before searching the other sources.

---

## How I used an AI agent to build this

**Tool:** Claude, working as an agent with access to my repo, a terminal and the course page.

**What I asked for:** I pointed it at the assignment page and my fork and asked it to set the project up in my Documents/code folder and complete the assignment. The fork was still the empty Module 3 starter, so in practice that meant two jobs: finish the base recommender, then extend it into the Project 4 system. Along the way I steered it with follow-ups like "make it run without an API key", "every guardrail should tell the user what it changed", and "save the traces, don't just print them".

**What it generated or changed:**

- `data/songs.csv` - 10 new songs across 10 new genres
- `src/recommender.py` - loading, scoring, ranking, artist penalty, the `Recommender` class
- `src/main.py` - fixed the broken import (`from recommender import` failed when run as a module) and added five profiles
- New files: `src/retriever.py`, `src/planner.py`, `src/llm.py`, `src/guardrails.py`, `src/agent.py`, `src/vibefinder.py`, `src/formatting.py`, `knowledge/*.md`, `eval/run_eval.py`, `tests/test_recommender_extra.py`, `tests/test_ai_layer.py`, `diagrams/architecture.mmd`
- Ran the tests and the eval after each change and saved the outputs into `assets/`

**What I verified or fixed by hand, and where the agent got things wrong:**

- **The negation bug.** "road trip music, no country, nothing too sad" parsed "country nothing" as one phrase, so "sad" was never handled. The test caught it. The fix was a lookahead in the regex so a captured phrase can't run into the next negation word.
- **Retrieval leaking.** "more like Library Rain" pulled in the rainy day note. I only noticed because I read the "Based on these notes" section instead of just the song list.
- **Weak secondary notes.** "late night drive" matched both the late night note and the road trip note (via "drive"), and the road trip note was adding a high positivity target that made no sense for a moody night drive. Now a second note only fills gaps if it scored at least 80% as well as the best one.
- **Clamping vs dropping.** The first guardrail clamped every out-of-range number. I changed it so wildly out-of-range values (energy 7 on a 0-1 scale) get dropped, since clamping would turn a guess into "maximum energy".
- **Fake confidence.** "music that's not sad" came back at 0.97 confidence because lots of songs perfectly match a single positivity target. That's not real confidence, so single-preference requests now cap at 0.50 with a note asking for more detail.
- **Crisis handling.** The first version of the crisis check added the support message but still served the saddest songs in the catalog. I had it steer those requests toward calm songs instead.
- **The Gemini path is not verified against the real API.** I didn't have a key, so it's only tested with fake model functions. That's the least trustworthy part of the code and I've said so in the README.

**Design suggestions I pushed back on** are written up in the model card (section 9). The main ones: I didn't use embeddings and a vector store for retrieval because it was too much machinery for 30 notes, and I made the LLM optional instead of central so the eval could be deterministic.
