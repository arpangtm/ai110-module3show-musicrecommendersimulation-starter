# Model Card: VibeFinder 2.0

## 1. Model Name

**VibeFinder 2.0.** Version 1 was the plain Module 3 scorer that took a preference dictionary. Version 2 is the applied AI system that takes a sentence.

---

## 2. Intended Use

VibeFinder suggests a handful of songs from a 20-song catalog based on a short plain English request, like "something for a rainy day" or "gym music but no metal". It explains every pick and says how confident it is.

It's a classroom project. It's meant for exploring how recommenders turn fuzzy human requests into numbers and where that goes wrong, not for real listeners. The catalog is fictional, so it can't actually help you find music to play.

Assumptions it makes about the user: that they're describing one listening situation at a time, that they're writing in English, and that what they want can be expressed with genre, mood, energy, acousticness, tempo and positivity. That last one is a big assumption and most of the limitations below come from it.

---

## 3. How the Model Works

Imagine you walk into a record store and say "I need something for studying." A good clerk doesn't just search the word "studying". They know from experience that people studying usually want something calm, steady and not too wordy, and they go grab a few things that fit.

VibeFinder tries to do that in four steps.

First, it looks up notes. I wrote a small set of notes about common situations (studying, gym, rainy day, sleep, date night and so on) and about each genre, including the other names people use ("rap" for hip hop, "lo-fi" for lofi). It also checks whether you mentioned a song by name. Each note comes with suggested settings, so the studying note says roughly "lofi, focused, low energy, acoustic is good."

Second, it builds a profile from those notes and anything you said directly. If you said "slow" or "acoustic" or "top 3", those win over the notes. If you said "no metal", metal gets excluded.

Third, it scores every song. A song gets points for matching your genre (2 points), your mood (1.5), being close to your energy level (up to 2), fitting your acoustic preference (up to 1), and smaller amounts for tempo and positivity. If an artist already appears in the list, their next song loses a point so one artist doesn't take over.

Fourth, it checks its own work. It looks at how close the top songs came to a perfect score, whether any of them have the mood you asked for, and whether the energy is way off. If something's wrong it tries again with a different balance (caring more about mood, or more about energy), then keeps whichever attempt had the fewest problems. If none of them were good, it tells you.

**Changes from the starter logic:** the starter had no scoring at all, so all of this is new, but the key decisions were: energy is scored by closeness instead of "higher is better" (a sleep playlist shouldn't reward loud songs), "indie pop" gets half credit for someone who asked for "pop", ties are broken alphabetically so the output is stable, and the artist penalty.

---

## 4. Data

The catalog is 20 songs in `data/songs.csv`. The starter came with 10 (mostly pop, lofi and a few others); I added 10 more to cover hip hop, EDM, classical, country, metal, R&B, folk, reggae, latin and blues. That gives 17 genres and 15 moods, but most genres have exactly one song. Lofi has three, pop has two.

Each song has genre, mood, energy, tempo, valence (how positive it sounds), danceability and acousticness. I made the numbers up to be believable for each style. They aren't measured from real audio.

The knowledge base is two markdown files I wrote: 16 situation notes and 17 genre notes.

What's missing: lyrics, language, artist background, release year, popularity, anything about the listener's history, and most of the world's music. There's no k-pop, no Afrobeats, no gospel, no Bollywood, no regional Mexican music. The moods are one tag per song, so a song can't be both sad and danceable even though tons of songs are.

---

## 5. Strengths

It does well when a request maps to one clear situation or genre. "Chill lofi for studying", "music for meditation", "romantic dinner date" and "I'm so angry I need to vent" all put a genuinely fitting song at the top with high confidence.

The explanations are honest. Every number in the "Why" column comes straight from the scoring rules, so you can check it with a calculator. That matched my intuition way more than I expected; when a result looked wrong, the reasons almost always showed me exactly why (usually energy carrying a song that matched nothing else).

It's good at saying no. Empty input, off-topic questions and impossible combinations all get a clear message instead of a confident wrong answer.

---

## 6. Limitations and Bias

**The catalog is the biggest bias.** With one song in most genres, someone who likes jazz, hip hop or metal gets one real match and four songs from other genres, while a lofi or pop listener gets a list that actually feels like their genre. The system effectively serves some tastes better than others just because of what's in the CSV. In a real product the same thing happens with popular vs niche genres, and it compounds because the popular genres also get more listening data.

**Energy dominates once genre and mood run out.** Energy is worth up to 2 points and every song gets some of them, so after the first one or two picks the list is basically sorted by "how close is the loudness." That's why a gym profile asking for EDM gets a metal song second.

**Mood labels are my opinion.** I decided what counts as "melancholy" or "romantic". Another person would tag some songs differently, and the system would recommend differently.

**The situation notes encode stereotypes.** "Romantic means R&B and latin", "angry means metal", "nostalgic means country". Those are common associations, not facts, and they push certain genres into certain emotional boxes. Someone whose idea of a romantic song is a metal ballad gets nothing from this.

**English only, and only the words I thought of.** Slang, other languages and anything I didn't put in a keyword list falls through. People who describe music differently from me get worse results.

**Single preference profile.** It can't hold "mostly chill but a couple of upbeat ones", and it can only weight one genre at a time.

---

## 7. Evaluation

I tested in three ways.

**Fixed profiles on the base recommender.** I ran five hand-written profiles (pop, gym, study, sad acoustic, rap). I was checking whether the top song was the obvious one and whether the reasons made sense. Each profile got the right number one. The surprise was how fast quality dropped after the first slot, and seeing metal show up in the gym list.

**An automated test suite.** 36 pytest tests, all passing. They check loading (including a CSV with a broken row), score bounds for every song, sorting, the diversity rule, retrieval from each source, every guardrail, end-to-end agent runs, and the LLM path using fake models that return clean JSON, chatty prose, garbage and exceptions.

**An evaluation script** (`python -m eval.run_eval`). 17 requests with a written expectation for each, run twice to check the answer doesn't change between runs.

```
17/17 checks passed | 17/17 gave identical results on a second run | average confidence on answered requests: 0.71
```

| Input | Expected | Result |
|---|---|---|
| chill lofi for studying | lofi on top, high confidence | Pass, 0.86 |
| something to listen to on a rainy day | melancholy on top (only possible via retrieval) | Pass, 0.85 |
| gym playlist, no metal | no metal, energetic top song | Pass, 0.77 |
| kpop for a party | warns kpop is missing, still energetic | Pass, 0.76 |
| (empty) | refuse | Pass |
| what's the capital of France | refuse | Pass |
| slow happy songs | detect conflict, revise, warn | Pass, 0.66 |
| acoustic metal | admit low confidence | Pass, 0.44 |
| sad songs, honestly I want to die | support message first | Pass |

The full table is in `assets/eval_report.md`. The same script also runs five requests with retrieval turned off: four out of five get refused without it, and all five get answered with it.

To be honest about what "17/17" means: I wrote the expectations, and I wrote them knowing roughly what the system does. The checks prove the behavior I designed is there and stays there; they don't prove a real listener would like the songs.

---

## 8. Future Work

- A bigger catalog with real audio features (a public dataset with a few thousand tracks), and at least a few songs per genre so the bias above shrinks.
- Multiple moods per song, and letting profiles hold a mix ("mostly chill, some upbeat").
- Embedding-based retrieval so synonyms work without me listing them.
- A feedback loop: thumbs up and down that nudge the weights for that listener, which is the first step toward the collaborative-filtering half of real recommenders.
- A better diversity rule that also spreads genres, not just artists.
- Run the eval against the real Gemini planner and compare it to the rule planner on the same 17 cases.

---

## 9. Reflection and Ethics

### What are the limitations or biases in your system?

The short version is in section 6. The one I'd put first is that the catalog decides who gets good recommendations. A listener with mainstream taste gets five songs that feel like theirs. A listener who likes jazz gets one and four guesses. Nothing in the scoring is unfair on its own; the unfairness comes entirely from what data exists. I think that's the most realistic thing about this project, because real platforms have the same problem at a much bigger scale. The second one is that my situation notes bake in stereotypes about which genres fit which feelings, and the system trusts those notes completely.

### Could your AI be misused, and how would you prevent that?

Honestly, a 20-song fictional recommender isn't very dangerous. The realistic risks are smaller:

- **Prompt injection.** If the Gemini planner is on, someone could try to put instructions in their request to get the model to do something else. I handle that in layers: obvious injection phrases get stripped and flagged, the request is wrapped in tags and the prompt tells the model to treat it as data, the model is only allowed to return a JSON profile, and that profile goes through the same validation as the rule planner. Even if the model got fully hijacked, the worst it could do is return a weird profile, and anything outside the catalog or out of range gets dropped. It never gets to rank songs or write the response text.
- **Someone in distress.** A person might ask for sad music while saying something that suggests they're thinking about hurting themselves. Matching that mood with the saddest songs in the catalog would be the wrong call. VibeFinder checks for that kind of language, puts a support message with the 988 line at the top, and shifts the list to calm songs instead. It's keyword based so it will miss things, but it's better than pretending music is the only thing that matters in that moment.
- **Overtrust.** People tend to believe confident output. That's why the confidence score exists, why it gets capped when the request is thin, and why weak filler songs get removed rather than padded in.

If this were real, I'd also want rate limiting on the LLM path and a way for users to report bad recommendations.

### What surprised you while testing your AI's reliability?

Three things.

First, the bug I didn't know I had. I wrote a test for "road trip music, no country, nothing too sad" assuming it would pass, and it failed. My negation regex grabbed "country nothing" as one phrase, so "sad" never got handled. Nothing crashed and the output looked reasonable, which is exactly why I wouldn't have caught it by just running examples.

Second, "acoustic metal" taught me that confidence needs to be about the whole list, not the top pick. The one metal song in the catalog is a perfect genre match, so the top result looks great, but the rest of the list is just quiet songs that happen to be acoustic. Averaging the top three is what made the system admit it was struggling.

Third, re-weighting doesn't fix impossible requests. I expected the self-critique loop to be the hero. For "slow happy songs" it tried all three strategies and each one just traded one problem for another, because there's no slow happy song in the catalog. The useful part ended up being that the loop makes the system notice the conflict and say so, not that it finds a better list.

### Collaboration with AI during this project

I used Claude as a coding partner the whole way through: talking through the architecture before writing anything, writing first drafts of modules, generating test cases, and debugging when the output looked off. My process was usually to describe what I wanted a piece to do and why, read what came back, run it on real inputs, and then push back on anything that looked wrong or overbuilt.

It was most useful for the boring-but-important parts. It wrote the guardrail validation and most of the pytest cases faster than I would have, and when I asked it to try to break the system, it came up with inputs I hadn't thought of, like an LLM returning `energy: 7` because it was thinking on a 1-10 scale.

**A suggestion I accepted:** making the LLM optional and having a rule-based planner as the default, with the LLM's output going through the same validator. My first instinct was to make Gemini the main thing because that's what "applied AI" sounded like to me. The argument that convinced me was about evaluation: if the core path depends on an API, the results change between runs and anyone without a key can't run it, so I couldn't write a real pass/fail eval. It was right. The deterministic eval ended up being the thing that caught real bugs.

**A suggestion I changed:** for retrieval, the first suggestion was to use sentence embeddings with a vector store, which is the standard RAG setup. I didn't take that as written. For about 30 short notes it was way more machinery than the problem needed, it would add a heavy dependency and a model download, and it would make the retrieval harder to explain. I went with keyword matching plus a small word-overlap bonus, where the trace shows exactly which words matched. It's less clever, but I can debug it in ten seconds.

**Another one I rejected:** for the profile validator, the first version clamped every out-of-range number into range. I changed that so values that are wildly out of range (like energy 7 on a 0-1 scale) get dropped instead of clamped, because clamping 7 to 1.0 would turn "medium energy on a 1-10 scale" into "maximum energy", which is worse than ignoring it.

Overall the AI was fast and good at generating options, but it tended to reach for the bigger, more standard solution. Most of my actual decisions were about scaling things back down to what this project needed.

### Personal reflection

Before this I thought of recommenders as mostly a math problem: get the weights right and the songs fall out. Building this changed that. The weights barely mattered with 20 songs; the interesting work was all in interpretation (what does "rainy day" mean as numbers?) and in deciding how the system should behave when it can't do what was asked. I also look at apps like Spotify a bit differently now. When it recommends the same few artists over and over, I wonder how much of that is my taste and how much is just what the catalog and the data make easy to recommend.
