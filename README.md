# Reasoning Interruption in LLMs: A Null Result

**TL;DR.** I tested a common assumption (that making an LLM "stop and
think" mid-reasoning improves its performance across) five interruption
strategies, three benchmarks, and two models (4,500 model calls). I found
**no reliable effect**. Of 24 condition comparisons, only 3 reached
significance, scattered across different models, benchmarks, and
directions, which is consistent with chance. The more useful finding was
methodological: a bug in how answers were extracted from free-text
responses silently manufactured an apparent effect that disappeared once
extraction was done correctly. Both stories are below.

---

## Motivation

This project tests a common assumption in the AI space: that making an
LLM "stop and think" mid-reasoning improves its performance on reasoning tasks.

I was drawn to this by the implications of a *negative* result. If
forcing this behaviour doesn't help, it's a warning against a deeper
habit: personifying AI by assuming that traits intrinsic to human
cognition will transfer to artificial neural networks. The human
tendency to pause, second-guess, and reason non-linearly feels like it
should help, but that intuition comes from how *we* think, not from how
these models work. So the result speaks to two things at once: a
practical caution against reflexively telling models to "think again,"
and a broader caution against attaching human cognitive characteristics
to systems that don't share our architecture.

## Method

**Five conditions**, each a prompt prefix on top of chain-of-thought:

| Condition | Name | Instruction |
|-----------|------|-------------|
| A | Baseline | Standard step-by-step chain-of-thought |
| B | Scheduled | Stop every 2 steps and ask "am I missing another angle?" |
| C | Random | Pause and reflect at a random point mid-reasoning |
| D | End-review | Review the full reasoning chain before the final answer |
| E | Signal-driven | Pause when choosing between equally plausible paths, switching domains, or relying on an unverified assumption |

**Three benchmarks:** StrategyQA (yes/no, insight-style multi-hop),
ARC-Challenge (multiple choice, science), GSM8K (numeric, procedural math).

**Two models:** Claude Haiku and GPT-4o-mini.

**Scale:** 150 questions per condition per benchmark per model. Primary
metric is accuracy; a secondary model-graded reasoning-quality score was
also collected.

## Results

Accuracy by condition (after the extraction fix described below):

**Claude Haiku**

| Cond | StrategyQA | ARC | GSM8K |
|------|-----------|-----|-------|
| A (baseline) | 66.7% | 92.0% | 94.7% |
| B (scheduled) | 63.3% | 90.0% | 92.7% |
| C (random) | 73.3% | 91.3% | 97.3% |
| D (end-review) | 72.0% | 94.7% | 96.7% |
| E (signal-driven) | 62.7% | 90.0% | 87.3% |

**GPT-4o-mini**

| Cond | StrategyQA | ARC | GSM8K |
|------|-----------|-----|-------|
| A (baseline) | 68.0% | 86.0% | 95.3% |
| B (scheduled) | 60.0% | 88.7% | 94.7% |
| C (random) | 68.0% | 88.7% | 96.0% |
| D (end-review) | 67.3% | 90.7% | 95.3% |
| E (signal-driven) | 63.3% | 86.0% | 97.3% |

**Significance (McNemar's test, each condition vs. baseline A).** Of 24
comparisons, 3 reached p < 0.05: Haiku GSM8K/E (p=0.027), GPT-4o-mini
StrategyQA/B (p=0.017), and GPT-4o-mini ARC/D (p=0.016, the only
*positive* one). They're scattered across different models, benchmarks,
and directions with no consistent pattern. Across 24 tests, roughly one
false positive at p<0.05 is expected by chance, so 3 unrelated hits is
barely distinguishable from noise.

**Interpretation.** No interruption strategy produced a consistent,
reproducible change in accuracy. There's a visible numerical lean toward
the random (C) and end-review (D) conditions on Haiku StrategyQA, but it
doesn't reach significance (C vs. A: p=0.099) and doesn't replicate on
GPT-4o-mini, so I don't claim it as an effect. ARC and GSM8K were
near-saturated (90%+) for every condition, leaving little room to detect
differences with StrategyQA being effectively the only benchmark where
conditions could separate.

**Bottom line:** no evidence that prompting-level reasoning interruption
reliably helps or harms accuracy at this scale which is consistent with the
motivation above.

## The Extraction Problem

The most important issue in this project was a bug in how answers were 
extracted from the models' free-text responses.

The benchmarks need a committed answer (a letter, or yes/no) pulled out
of a paragraph of reasoning. My initial extractor grabbed the *first*
answer in the response, i.e the one stated before the model interrupted
itself and potentially changed its mind. This silently skewed the
results: it affected the very thing the experiment was testing.

Condition A (plain reasoning) was barely affected, because it tends to
answer straight without backtracking. But the self-interrupting
conditions which are *designed* to make the model reconsider and often
change its answer, were affected heavily. The extractor kept recording their
discarded first answers instead of their final ones. This made the
interruption conditions look worse than they actually were, and produced
a false trend: it looked like over-prescriptive interruption hurt
performance, when in reality it didn't.

I caught this by spot-checking results from the pilot run, pulling a few
"incorrect" responses and reading them by hand, where it became obvious
the model had actually answered correctly. After re-extracting answers
properly (a model-based pass that reads each response for its *final*
committed answer), a superficially significant finding turned out to be
non-significant. Concretely: with the buggy parser, conditions B and E
significantly underperformed baseline on Haiku StrategyQA (p=0.013); after
the fix, those same comparisons were non-significant (B: p=0.332, E:
p=0.210).

The lesson, for anyone running a similar study: don't trust your summary
numbers until you've checked them against raw outputs. Build failsafes
into your extraction, and rigorously verify pilot results before scaling
up, as a measurement error that correlates with your experimental condition
can manufacture a result that looks real.

## Limitations

- **Underpowered.** 150 questions per cell can't detect small effects; the
  null is really "no large effect."
- **Saturated benchmarks.** ARC and GSM8K sat above 90% for all conditions,
  so only StrategyQA was a live test.
- **Single run.** No repeated sampling; temperature and seed effects not
  characterized.
- **Two models**, both in a similar capability tier; larger models may behave
  differently.

A stronger follow-up would use larger samples, harder/unsaturated
benchmarks, and an answer format that removes the extraction problem by
construction (e.g. a forced final-answer line).

## Repo Structure

- `Experiment.py` : main experiment runner (conditions × benchmarks, Anthropic)
- `real_experiment_openai.py` : same harness, GPT-4o-mini
- `rescore_arc_model.py` : model-based answer extraction (the fix)
- `significance.py` : McNemar tests vs. baseline
- `qualitative.py` : pulls cases for manual inspection
- `load_benchmarks.py` : dataset loading
- `Summarise_results.py` : results summary
- `real_experiment_haiku_fixed.csv` / `real_experiment_openai_fixed.csv` : corrected results
- `real_experiment_haiku.csv` / `real_experiment_openai.csv` : raw run outputs, kept for transparency

## Reproducing

```bash
pip3 install anthropic openai datasets
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
python3 Experiment.py
python3 rescore_arc_model.py
python3 significance.py
```

Roughly \$7–10 (USD) in API usage for a full two-model run.

---

*Student research project. The headline finding is a null result; its
value is (1) a documented test of a widely-held prompting intuition and
(2) a worked example of how answer-extraction artifacts can fabricate
apparent effects in prompting studies.*
