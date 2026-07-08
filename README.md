# AI Email Reply Suggester

A tool that suggests replies to incoming emails. Given a subject and body, it
retrieves similar past emails from a small reference dataset, uses them as
examples to generate a grounded reply, and scores that reply against an
explicit, weighted rubric using an LLM-as-judge evaluator. The judge itself
is validated against a hand-labeled sample, an adversarial test set, and a
consistency check.

## How It Works

### 1. Retrieval

Every row in the reference dataset is embedded locally with
`sentence-transformers/all-MiniLM-L6-v2` (Groq does not offer an embeddings
endpoint). For a new email, the three most similar past emails are retrieved
by cosine similarity. See `generator/retrieve.py`.

### 2. Generation

The retrieved examples are injected into the prompt as few-shot context,
alongside instructions covering tone, structure, and a directive not to
invent facts. This approach was chosen over plain prompting (too generic
without dataset grounding) and fine-tuning (impractical at this dataset
size). See `generator/prompt_templates.py` and `generator/generate_reply.py`.

### 3. Evaluation

Each generated reply is scored by a second LLM acting as a judge, against
four weighted dimensions:

| Dimension | Weight | Question |
|---|---|---|
| Relevance / task completion | 0.40 | Does the reply address what was asked? |
| Factual grounding | 0.25 | Does it avoid inventing facts, dates, or commitments? |
| Tone / style fit | 0.15 | Does it match the expected register? |
| Completeness | 0.20 | Are all parts of the request covered? |

Relevance carries the highest weight, since a well-written reply to the
wrong question has little value. String overlap and semantic similarity to
a "correct" reply are deliberately excluded from this score — a past reply
is one valid answer, not a unique ground truth — and are reported only as
supporting signals. See `evaluator/rubric.py`.

The judge runs on a different Groq model than the generator
(`JUDGE_MODEL` vs. `GEN_MODEL`, configured in `.env`) to reduce, though not
eliminate, the risk of a model grading its own output. Note that the
*generator* uses the same model as dataset generation; only the judge is a
separate model.

**Reference-guided vs. blind judging.** When scoring a dataset row with a
known past reply, the judge is shown that reply as reference context,
explicitly framed as "one valid answer, not the only correct one" rather
than ground truth (see `evaluator/judge.py`). This makes the full-dataset
score below reference-guided rather than blind. New, ad-hoc emails (via
`scripts/run_end_to_end.py`) and the adversarial/consistency checks below
have no such reference and are judged blind — the two sets of numbers are
not directly comparable.

## Results

### Full-Dataset Evaluation

Run across all 45 dataset rows, with each row's own reply excluded from its
retrieval pool to prevent leakage:

| Metric | Value |
|---|---|
| Mean | 91.05 / 100 |
| Min | 62.5 |
| Max | 100.0 |
| Std. dev. | 11.32 |

By category: `scheduling` 94.86, `complaint` 92.37, `customer_support`
90.38, `sales` 88.24, `internal` 87.74.

A few dataset rows describe near-duplicate scenarios (see
[`dataset/README.md`](dataset/README.md)) and may retrieve a near-identical
past reply as an example, which can nudge those individual scores upward.

### Judge Validation

**Calibration.** A 12-response sample, drawn from all 5 categories, was
scored independently by a human labeler and compared against the judge
(`evaluator/calibration/human_labels.jsonl`,
`evaluator/calibration/validate_judge.py`):

```json
{
  "relevance":    { "n": 12, "pearson_r": 1.0,   "pct_within_1": 100.0 },
  "grounding":    { "n": 12, "pearson_r": 0.906, "pct_within_1": 100.0 },
  "tone":         { "n": 12, "pearson_r": 0.188, "pct_within_1": 91.7 },
  "completeness": { "n": 12, "pearson_r": 1.0,   "pct_within_1": 100.0 }
}
```

Relevance, grounding, and completeness show strong agreement. Tone's
correlation is low despite 91.7% practical agreement (within one point on
11 of 12 rows): with a sample this small and scores clustered at 4–5, a
single disagreement swings the correlation sharply. In the one outlier, the
judge rated a reply's generic-professional tone as a good fit for an email
labeled casual; the human labeler scored it lower for reading too formal.
This reads as a small-sample artifact rather than systematic judge
unreliability on tone — the near-universal within-one-point agreement is
the more informative signal here.

**Adversarial test.** Five hand-written replies — three deliberately bad
(off-topic, hallucinated commitment, rude tone) and two deliberately good
(direct answer, paraphrase) — were scored by the judge
(`evaluator/calibration/sanity_checks.py`):

```json
{
  "scores": {
    "bad_wrong_topic": 0.0,
    "bad_hallucinated_commitment": 42.5,
    "bad_rude_tone": 18.8,
    "good_direct_answer": 93.8,
    "good_paraphrase": 81.2
  },
  "separation_ok": true
}
```

The weakest good reply outscores the strongest bad reply by 38.7 points.

**Consistency test.** The same reply, judged twice:

```json
{
  "first":  { "scale_1_5": 4.5, "normalized_0_100": 87.5 },
  "second": { "scale_1_5": 4.5, "normalized_0_100": 87.5 },
  "diffs":  { "relevance": 0, "grounding": 0, "tone": 0, "completeness": 0 },
  "stable": true
}
```

### Supporting Metrics

Reported alongside the primary score but never folded into it:

- Embedding cosine similarity between the generated and the sent reply.
- Greeting and sign-off presence.
- Reply length relative to the category median.
- Entity-echo: whether names, dates, and order numbers from the incoming
  email appear in the reply.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# set GROQ_API_KEY in .env (create one at console.groq.com)
```

Groq's free tier enforces a daily token quota per account, which a
full-dataset run can exhaust. To configure automatic failover across
multiple keys, set `GROQ_API_KEYS` (comma-separated) instead of
`GROQ_API_KEY` — every call rotates to the next key on a rate-limit
response. See `.env.example` and `generator/config.py`.

### Usage

Generate a reply to a new email:

```bash
python -m generator.generate_reply --subject "Refund status?" --body "I returned item #4521 two weeks ago and haven't heard back."
```

Generate a reply and score it in one step:

```bash
python -m scripts.run_end_to_end --subject "Refund status?" --body "I returned item #4521 two weeks ago and haven't heard back."
```

Evaluate the full dataset and print an aggregate score:

```bash
python -m scripts.run_end_to_end --full-dataset
```

To persist per-row results to `results/eval_results.jsonl` (the source of
the calibration numbers above), use the evaluator CLI directly:

```bash
python -m evaluator.evaluate
```

### Tests

The test suite is fully mocked and requires no API key:

```bash
pytest
```

## Project Structure

```
dataset/    synthetic dataset and generation script
generator/  retrieval, prompting, and reply generation
evaluator/  rubric, heuristics, judge, calibration, and evaluation CLI
scripts/    end-to-end entry point
tests/      unit tests
```

## Limitations

- The dataset is synthetic; see `dataset/README.md` for details.
- The calibration sample is small (n = 12) and labeled by a single reviewer
  rather than blind or independent raters.
- The dataset-level headline score is reference-guided, as noted above.

## Development Notes

Claude Code was used to scaffold the project end to end — dataset
generation, retrieval, prompting, the judge, and calibration tooling —
starting from a written design spec. The rubric weights, judge model
choice, and calibration sample were reviewed, and the sanity checks were
run and inspected, before being accepted.
