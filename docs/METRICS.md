# Metrics, in plain words

Convention everywhere: **label 1 = the model's answer was wrong**, and every scorer outputs a
**risk** where higher means "more likely wrong".

| Metric | The question it answers | Good value |
|---|---|---|
| **AUROC** | Pick one wrong and one right answer at random: how often does the risk score rank the wrong one higher? | 0.5 = coin flip, 1.0 = perfect |
| **ΔAUROC** | How much one scorer beats another on the same questions | > 0, with a CI that excludes 0 |
| **Bootstrap 95% CI** | How much would this number move with a different sample of test questions? (1,000 resamples) | Narrow, and away from 0.5 / 0 |
| **Paired bootstrap** | Is scorer A *really* better than B? Both are scored on the same resamples, so per-question difficulty cancels out | CI excludes 0 → significant |
| **Significant seeds (e.g. 5/5)** | Did the result hold on how many of 5 different train/test splits? Covers noise the bootstrap can't: which questions landed in training | ≥ 3/5 **and** positive mean → "robust" |
| **Rank correlation** | Do the probe and the model's token confidence flag the *same* questions? | Near 1 = redundant; low = complementary |
| **Transfer AUROC** | A probe trained on trivia errors, tested on true/false statements it never saw: does it flag the false ones? | > 0.5; < 0.5 means *inverted* |
| **Selective accuracy @50 / @80** | If the model only answers the 50% / 80% of questions it's most confident about, how accurate is it? | Higher than overall accuracy |
| **AURC** | Selective accuracy summarised over every coverage level at once | Lower = better |
| **ECE** | Calibration: when the probe says "70% risk", is the answer wrong ~70% of the time? | Near 0 |
| **Forward passes** | How many times the model runs per question: the hardware-independent cost | Lower = cheaper |

## The comparisons in the scale table

- **Beyond difficulty:** the probe plus question features, vs question features alone. Do the
  hidden states add anything that the question text doesn't already reveal?
- **Model-specific:** this model's activations vs a *neighbouring* model's activations,
  predicting *this* model's errors. Another model can judge how hard a question is, but it
  can't see this model's knowledge. Winning means the signal is about this model itself.
- **Probe vs output confidence:** the probe vs the model's least-confident answer token.
- **Probe + output vs output:** combined (by stacking) vs token confidence alone. Winning means
  the probe catches errors that token confidence misses.

## Metric that turned out to be unreliable

**Cosine similarity between probe directions** (in `transfer.csv`). When a dataset is
perfectly separable (in-domain AUROC ≈ 1.0), many different directions separate it equally
well, so two working probes can point in unrelated directions. Treat transfer AUROC as the
evidence, not the cosine.
