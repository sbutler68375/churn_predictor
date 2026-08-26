# Model Metrics & Business Trade-offs

## Why recall matters more than precision here

A churn model for a B2B SaaS company is not graded on being "right" in the
abstract -- it's graded on the cost of being wrong in each direction, and
those costs are wildly asymmetric.

- **False positive** (flagging a healthy company as at-risk): a Customer
  Success manager spends a few minutes on an unnecessary check-in call. Mildly
  annoying, essentially free.
- **False negative** (missing a company that's actually about to churn): a
  real contract -- potentially a $100,000+ enterprise account -- quietly
  cancels with zero warning, because nothing ever told anyone to intervene.

Given that asymmetry, this system is deliberately tuned to prioritize
**recall** (catching as many real at-risk companies as possible) over
**precision** (minimizing false alarms). A model that occasionally sends a CS
manager to check on a healthy account is doing its job. A model that lets a
real churn risk slip through silently is not.

## A leakage bug we caught and fixed

The first version of the training features included `engagement_velocity` --
the exact same column used to define the `is_churned` label in the SQL view
(`is_churned = 1 WHEN engagement_velocity > 0.15`). Training on that column
produced a perfect 1.0 across every metric for all three models, which was
the tell: the models weren't predicting churn from behavior, they were just
reading the threshold rule back off a column that already encoded the answer.

Fix: `engagement_velocity` was removed from the model's input features. It
remains in the SQL view purely to define the label, not to predict it. Every
result below reflects the corrected feature set.

## Test-set results (single 80/20 split, 20 held-out companies)

| Model | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.750 | 0.857 | 1.000 |
| Random Forest | 1.000 | 0.750 | 0.857 | 1.000 |
| XGBoost | 1.000 | 0.750 | 0.857 | 0.984 |

Zero false alarms on this split; 3 of 4 truly at-risk companies caught, 1
missed.

## 5-fold stratified cross-validation (mean ± std, full 100-row dataset)

| Model | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.950 ± 0.100 | 0.900 ± 0.200 | 0.905 ± 0.131 | 0.992 ± 0.016 |
| Random Forest | 0.893 ± 0.137 | 0.833 ± 0.211 | 0.844 ± 0.151 | 0.993 ± 0.009 |
| XGBoost | 0.867 ± 0.163 | 0.783 ± 0.194 | 0.798 ± 0.126 | 0.942 ± 0.072 |

## Honest limitations

- **Small sample size.** The dataset has only 100 companies, ~19-20 of them
  at-risk. Each cross-validation fold's test slice contains just 3-4 at-risk
  companies, so a single miss or catch swings recall by 25 points. The large
  standard deviations above are a direct, expected consequence of dataset
  size, not evidence of an unstable model.
- **ROC AUC is the most trustworthy number in this table.** It evaluates
  ranking quality across every possible threshold rather than one fixed
  cutoff, so it's far less sensitive to the small-sample noise described
  above. All three models sit at 0.94-0.99, which is the more reliable signal
  of real separation between healthy and at-risk behavior.
- **Simpler beat more complex, here.** Logistic Regression posted the
  strongest and most stable scores of the three, edging out Random Forest and
  XGBoost. That's not the usual outcome, but it's not surprising on a small
  dataset (80 training rows) with a clean, strongly-separated signal --
  XGBoost's extra flexibility has more room to overfit when there isn't much
  data to constrain it.
- **Synthetic data.** The underlying churn pattern was deliberately injected
  into the data generator (Week 1), with a sharp behavioral break between
  healthy and at-risk cohorts. Real-world churn signals are almost always
  noisier and more gradual than this, so these metrics should be read as
  "the pipeline correctly recovers a known synthetic signal," not as a claim
  about real-world churn prediction accuracy.

## Model selection for explainability (Week 4) and the dashboard (Week 5)

Logistic Regression scored best in cross-validation, but SHAP's
`TreeExplainer` -- the fast, exact explainer used in this project -- only
supports tree-based models. Rather than switching to a slower, approximate
explainer (e.g. `KernelExplainer`) just to keep the top CV scorer, **Random
Forest** was chosen as the model carried forward: it is tree-based (clean
`TreeExplainer` support), and its cross-validated scores (precision 0.893,
recall 0.833, ROC AUC 0.993) are close behind Logistic Regression and
noticeably more stable than XGBoost on this dataset. XGBoost was set aside
for this project -- it had the weakest and noisiest cross-validation results
of the three, likely due to overfitting on only 80 training rows.
