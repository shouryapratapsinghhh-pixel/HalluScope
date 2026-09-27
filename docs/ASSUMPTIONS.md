# Assumptions, deviations, and mistakes caught

## Assumptions

- **Grading:** an answer is correct if any normalized accepted alias appears in the
  normalized answer (containment, not exact match). Small models wrap answers ("It was
  Paris, France"), so exact match would be unfairly strict. Some false positives are possible.
- **Closed-book, fixed prompt:** `Question: ...\nAnswer:`, no few-shot examples, generation
  stopped at the first newline.
- **Question-difficulty control:** surface features of the question text only (length, digits,
  proper-noun count, question word, ...). A richer control (e.g. an embedding from a
  separate encoder) could explain more of the probe's signal.

## Deviations from the original plan

- **Hypothesis "the signal lives in middle layers": refuted.** The error signal peaks in the
  upper layers; the mid layers held a mismatched-association detector.
- **No 3B model:** Qwen2.5-3B stalled on the very first question on this laptop (memory).
  The scale study used the Pythia suite instead, which was trained identically across sizes.
  That's arguably a cleaner scale comparison.
- **Precision:** early caches ran in float32, later ones in float16 on Apple Silicon. The
  greedy answers matched the cache 100% in the self-consistency run, so the effect on labels
  appears negligible.
- **Not built:** verbalized-confidence baseline (small base models state confidence poorly),
  MLP probe (proves less than a linear one), cross-model probe transfer.

## Mistakes caught along the way (and fixed)

1. **Crude combination of probe + other features.** Concatenating hundreds of activation
   dimensions with a few side features under one regularization penalty drowned the side
   features and gave unstable results. Replaced with stacking on out-of-fold probe scores.
2. **A flawed synthetic test.** A test world with errors defined as "a OR b" can't be
   represented by a logistic combiner, and it left only ~40 negatives in the test split.
   Rebuilt with additive causes and balanced classes.
3. **An early result inflated by one lucky split.** Pythia-410M's first single-split probe
   AUROC was its best of five splits. This is why every result here is reported over 5 splits.
4. **A borderline result that was a sample-size artifact.** Qwen2.5-0.5B's beyond-difficulty
   signal was borderline at 1,000 questions and clearly robust at 3,000.
5. **An over-read of the mid layers.** Near-perfect transfer to true/false statements in the
   mid layers was first read as a "general truth representation". The negation control
   showed those layers invert under negation: they detect mismatched associations, not truth.
6. **An uninformative metric.** The cosine between probe directions can't be interpreted when
   data is perfectly separable (see METRICS.md).
