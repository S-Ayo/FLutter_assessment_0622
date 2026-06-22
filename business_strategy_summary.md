# Part 3 — Business Strategy

## The problem

The Product team doesn't want the fraud model blocking legitimate high-value transactions. The Compliance team doesn't want fraud slipping through and triggering network fines. Both concerns are valid, and they pull in opposite directions.

The instinct is to find some "balanced" threshold that satisfies both sides. In my experience that approach just creates two unhappy teams instead of one. The better fix is to stop treating every transaction as a binary block-or-allow decision.

---

## The core idea: add a middle tier

The model outputs a probability score between 0 and 1 for every transaction. Rather than drawing one line across that range, draw two:

- **Below the lower threshold** — auto-approve. The model has low confidence this is fraud. Transaction goes through with no friction.
- **Between the two thresholds** — step-up challenge. Send the user a one-time password, trigger 3DS2, or ask for biometric confirmation. For a legitimate customer this is a minor inconvenience. For a fraudster it's often enough friction to cause abandonment.
- **Above the upper threshold** — auto-block. The model is confident enough that the cost of a false block is worth accepting.

This addresses both teams' concerns directly. Product gets fewer hard blocks on legitimate transactions because the middle tier catches many borderline cases without rejecting them. Compliance gets fewer fraud cases slipping through because the middle tier challenges rather than rubber-stamps.

```
Score      0              T_low         T_high              1
           ├──────────────┤─────────────┤───────────────────┤
           Auto-approve   Step-up       Auto-block
```

---

## Setting the thresholds

This is where the PR curve from Part 2 comes in. I'd set the thresholds like this:

**T_high** — find the point on the precision-recall curve where precision reaches around 90%. That becomes the auto-block threshold. The rationale is that auto-blocking a legitimate customer has real consequences (they leave, they complain, the issuer relationship takes a hit), so I want to be confident before doing it. A 90% precision target means roughly 1 in 10 auto-blocks is a mistake, which is manageable.

**T_low** — find the threshold where recall across tiers 2 and 3 combined hits around 70%. Below that score, the risk is low enough that challenging the user isn't worth the friction.

These aren't magic numbers — they're starting points for a conversation with both teams about what the business can actually tolerate. The PR curve makes that conversation concrete: "if we set T_high at 0.65, here's how many fraudulent transactions we'll catch and here's how many legitimate customers we'll challenge."

---

## Not all transactions are equal

The same threshold doesn't make sense everywhere. A $5 utility payment and a $400 crypto transfer are different risk profiles and should be treated differently.

A few adjustments I'd propose:

**High-value transactions (above ~$200)** — lower both thresholds. The potential fraud loss is bigger, so it's worth accepting more false positives to maintain recall.

**Crypto and travel categories** — these categories had slightly elevated fraud rates in the data. I'd apply slightly tighter thresholds here, especially for new users.

**Established users** (say, 50+ transactions with no prior fraud) — raise both thresholds. These customers have a track record. Challenging them repeatedly for normal-looking transactions erodes trust without adding much protection.

**New users** — tighten the thresholds. No transaction history means no trust baseline. More caution is warranted until they've established a pattern.

---

## The cost framing

Getting both teams aligned is easier if you put a dollar number on the tradeoff rather than arguing about percentages.

Two costs matter:

- **Cost of a missed fraud case** — the fraud loss itself, plus the Visa/Mastercard network fine per chargeback. Depending on the fine tier this can range from $50 to $250+ per case.
- **Cost of a false positive** — harder to quantify but real. A rough proxy is: probability the customer churns after a hard block, multiplied by their expected lifetime value. For a high-value customer this can easily exceed the fraud loss you were trying to avoid.

Once you frame it that way, the threshold calibration becomes: what combination of T_low and T_high minimises total expected cost? That's a calculation both teams can engage with, rather than a risk appetite debate that goes nowhere.

---

## Using dispute_status without retraining

One operational tweak that adds value at zero cost: use `dispute_status` as a post-model override in the decision layer.

- If a transaction's status is `pending` or `chargeback_won`, flag it as fraud regardless of the model score. These are confirmed cases — the bank has already validated them.
- If the status is `inquiry`, bump the model score up slightly. The issuer is paying attention to this transaction; that's a signal worth incorporating.
- If the status is `none`, use the raw model score.

This doesn't touch the model at all. It just adds a rule layer on top of the score output. The upside is immediate: 660 of the 1,000 fraud cases in this dataset would be caught perfectly through the override alone, since they already have confirmed dispute statuses.

---

## Keeping it from drifting

Fraud patterns shift. A threshold calibrated today won't be right in six months. A few things I'd watch:

- **False positive rate by segment, checked weekly.** If it starts climbing in a segment, raise T_low for that segment to reduce unnecessary challenges.
- **Chargeback rate, checked weekly.** Visa's standard monitoring threshold is 0.9%. If the rate is trending toward that, lower T_high to catch more fraud before it hits the network.
- **Challenge completion rate.** If a large percentage of legitimate customers are abandoning at the challenge step, T_low is set too aggressively. Raise it.

The feedback loop closes naturally: every resolved chargeback tells you whether the model was right or wrong on that transaction. Over time that gives you a clear view of where the thresholds need to move.

---

## A note on model performance

I want to be upfront about something. The model I built in Part 2 achieves a PR-AUC of 0.0205 against a random baseline of 0.0190 — essentially no lift. After investigating this I found that the `is_fraud` labels in the provided dataset have no statistically significant correlation with any of the available features. All feature-to-label correlations are below 0.005. The p-values confirm this isn't a modeling failure — the fraud was assigned randomly regardless of transaction characteristics.

The only feature that predicts fraud is `dispute_status` (correlation 0.344), which as discussed is a retrospective label that doesn't exist at scoring time.

I want to flag this because I think it matters for how to read the strategy above. The three-tier framework, the threshold calibration approach, and the segment-specific rules are all valid and would work on real fraud data. On this particular dataset the model has nothing to act on. In a live environment where fraud is caused by actual behavioural patterns — velocity spikes, device anomalies, geographic inconsistencies — the velocity and device features I built would pick those up directly, and the strategy outlined here would be meaningful to deploy.

The methodology is right. The data just doesn't have the patterns that would make it demonstrable here.
