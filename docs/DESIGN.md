# ZeroFile: design

**Neural Storm 2026 · Human Impact · FinTech Inclusion · HI-04 Alternative Credit Scoring for the Unbanked**
Solo build by Aayush Sharma (Thapar University). The name refers to the people it serves: applicants with a zero-length credit file.

## Goal

A bureau score cannot rank someone who has no bureau file. In a public Indian vehicle-loan dataset, half of all
borrowers are in that position. ZeroFile scores those people from the loan application alone, explains every
result, shows what would change it, and reports what the policy would do to a lender's book.

**Success means:** a public demo a judge can use in under a minute, one metric they remember, and evidence
that survives a hard question.

## Decisions from the data check

Measured on the real data before the design was fixed:

| Question | Finding | Decision |
|---|---|---|
| Is the thin-file group big enough? | 116,950 of 233,154 borrowers (50.2%) have no bureau history | The pitch stands |
| How should we validate? | Default rate drifts by month: Aug 21.6%, Sep 19.2%, Oct 23.5% | Split by time and test on October, never randomly |
| What benchmark is fair? | No-bureau default (24.1%) is close to all scored borrowers (23.0%), which is too easy to beat | Benchmark against bureau **low-risk** grades A to G (19.0%) |
| Do we need age? | Adds about 0.000 AUC on no-bureau applicants | Excluded from the model, kept for the audit |
| Do we need state? | Adds about 0.002 AUC | Excluded from the model, kept for the audit |
| Do we need ID-document flags? | Removing all five costs about 0.001 AUC, and keeping them took about 33 points off an example applicant who gave a voter ID instead of Aadhaar | Excluded. A score should not punish someone for which identity document they share |
| Pincode and employee code? | Pincode adds almost nothing. The employee code records who logged the loan, not anything about the applicant | Dropped when the data loads |
| Reported LTV or computed? | Same AUC (0.631) | Use `loan / vehicle price`, so the form and the data agree |
| Monotonic LTV? | No AUC cost | Higher LTV can never lower risk, which makes recourse trustworthy |

## Decisions from the official brief

The official PDF arrived after the first build. Comparing it with the build changed four things:

| The brief covers | Gap found | Decision |
|---|---|---|
| Alternative data such as utility bills and mobile usage | The real data has neither, and no public dataset links them to repayment | Add a second model (`worker.py`) on an open synthetic dataset, which the PDF itself suggests, and label it everywhere |
| Score and limit advice | No loan amount was suggested | Add a suggested loan range from the monotonic models |
| Rural and gig workers | The real data marks neither | The worker model covers them directly, and self-employed applicants stand in on real data |
| Fairness across gender, region and occupation | Neither dataset has gender | Audit what exists, and state the gap in the app, README and model card |

The PDF also suggests Home Credit Default Risk as a base. It stays out because Kaggle's rules limit that data to its own competition. The organisers suggesting it does not change those terms.

## Decisions on the worker model

Measured on the Alternate Credit Dataset before the design was fixed:

| Question | Finding | Decision |
|---|---|---|
| How is the label made? | `defaulted` is `default_probability` above a fixed cutoff | Keep that column out of training, and never present the AUC as real-world skill |
| Which signals matter? | Bills, UPI, rent, loan burden and income swing carry it. Surveys, e-commerce and demographics add nothing | Use 7 inputs, which also keeps the form short |
| Do wealth stand-ins help? | Dropping income level and transaction sizes raised AUC from 0.930 to 0.933 | Score habits, not wealth |
| Do monotonic constraints cost anything? | AUC rose slightly with them | Keep them, so recourse is trustworthy |
| How should bands work? | A group-level target put the approve cutoff at 43% personal risk, above the high-risk cutoff | Use fixed cutoffs on a person's own risk: 15% (the average borrower) and 50% |
| Is the 100-point scale usable? | This model is sharper, so scores ran past 900 | Give it its own scale, with a test that it stays in 300 to 900 |
| Are its loan suggestions sane? | It offered ₹5,00,000 to someone earning ₹24,000, because the labels barely punish large loans | Add a stated 40% of income affordability cap |
| Who gets shut out? | Half of rural borrowers lack UPI, and no-UPI approval is far lower | Flag every no-UPI result for a person, and publish the gap |

## Does tuning help?

No. A scan of 32 LightGBM settings on the real data moved test AUC only between 0.629 and 0.631, and a plain logistic regression reaches 0.626. The limit is the data, so the time went into the product instead.

## Architecture

```text
Kaggle CSV ──> zerofile/data.py ──> zerofile/features.py ──> zerofile/train.py ──> artifacts/
 (not committed)  validate,        encode brand,          LightGBM,         model.txt
                  drop private     dealer, branch         calibration,      meta.json
                  columns          risk by time           thresholds,       report.json
                                                          audit                 │
                                                                                ▼
                 Streamlit app.py <──────── zerofile/scoring.py <──────────────────┘
                 worker · vehicle loan · lender · evidence      score, reasons, loan
                 · fairness · limits                            range, recourse

Alternate Credit Dataset ──> zerofile/worker.py ──> artifacts/worker_* ──> zerofile/scoring.py
 (synthetic, not committed)  inputs, LightGBM, audit
```

The deployed app reads only `artifacts/`. No borrower row ever reaches the host.

## Folder structure

```text
app.py                 Streamlit entry point
zerofile/data.py       download, validation, time split
zerofile/features.py   vehicle-loan features and target encoding
zerofile/evaluate.py   metrics, policy curves, fairness audit
zerofile/train.py      vehicle-loan training run
zerofile/worker.py     worker model: data, training, audit
zerofile/altdata.py    simulated signals for the data-value table
zerofile/tune.py       settings scan
zerofile/scoring.py    score, reasons, loan range and recourse for both models
artifacts/             trained models and aggregate reports (committed)
tests/                 pytest suite, including headless app tests
docs/                  design, deployment guide, demo script, submission text, screenshots
MODEL_CARD.md          intended use and limits
```

## Stack

Python 3.11 · LightGBM 4.7.0 · pandas 3.1.0 · NumPy 2.4.6 · Streamlit 1.66.1 · Altair 6.3.0 for the app.
scikit-learn 1.9.1 and pytest 9.2.0 for training and testing only. Host: Streamlit Community Cloud (free).

Streamlit alone is enough for one model and a form. LightGBM trains in about a minute on a laptop CPU, and its
model file is small enough to commit, so cold starts stay short and there is no GPU or paid API to fail.

## Data pipeline

1. `python -m zerofile.data` downloads the public [L&T Vehicle Loan Default](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) dataset into the ignored `data/` folder.
2. Load, check the schema, derive LTV and age, and **drop date of birth, pincode, customer ID and employee code**.
3. Split by disbursal date: train 1 Aug to 23 Sep, validate 24 to 30 Sep, test on October.
4. Encode brand, dealer and branch as smoothed historic default rates. Training rows use out-of-fold values, and later periods only see what training saw.

## Model

- **ZeroFile model:** LightGBM on 6 non-bureau features (LTV, vehicle price, employment type, and the brand, dealer and branch repayment records), with early stopping on validation and a monotonic LTV constraint.
- **Calibration:** Platt scaling fitted on validation. It stays smooth for the policy slider and needs nothing but NumPy at runtime.
- **Bands:** the approve and high-risk cutoffs are **fixed on validation before October is scored**, so the headline is not tuned on the test month.
- **ZeroFile Score (300 to 900):** a linear map of the model's log-odds, where 100 points halve the odds of default. SHAP values add up exactly to the score, so the breakdown is arithmetic and not an approximation.
- **Recourse:** the smallest extra down payment, or the cheaper vehicle, that reaches the approve band. Only levers a person can act on are offered, never identity, dealer or age.

## Evaluation

- **Headline:** the share of no-bureau applicants approved, their default rate against the bureau low-risk benchmark, and rupees of credit unlocked.
- **Ranking:** AUC with bootstrap 95% CIs, KS, and default rate by risk decile.
- **Comparisons:** an LTV rule at the same approval rate, the raw bureau score on rated borrowers (paired bootstrap), and ZeroFile with bureau data added.
- **Calibration:** Brier score, expected calibration error, predicted against actual by month.
- **Fairness:** approval rate, default rate, and approval among people who repaid, by age band, employment type and region, with the four-fifths rule as the flag.

## Stand-out features

1. **Lender impact simulator:** a risk-appetite slider that shows people approved, rupees unlocked and the default rate on the real holdout, plus your own economics to find the most profitable approval level.
2. **Explainable decision with recourse:** a points breakdown that adds up exactly, plus "put down ₹X more and you reach approval".
3. **An honest fairness audit:** it shows where each model fails, measures a fix, and routes those cases to a person.

## Known limits

- Every row is a loan the lender already funded, so the results are a back-test and say nothing certain about applicants it rejected.
- These are thin-file borrowers, not fully unbanked people, and the data covers vehicle loans from 2018.
- The target is first-instalment default, a noisy signal, so AUC stays modest.
- October was inspected during feature selection. Those choices only removed features, but it is not a fully blind test.
- The dataset licence is unclear, so raw data is never committed and the source is credited.

## Build plan

| By (IST) | Checkpoint |
|---|---|
| 3 Nov 17:00 | Data check and design done, repo scaffolded |
| 3 Nov 19:00 | Pipeline and training produce `report.json` |
| 3 Nov 21:00 | Scoring and app working locally |
| 3 Nov 23:00 | Tests, clean-install check, README, model card, guides |
| 4 Nov 12:00 | Repo public and app deployed |
| 4 Nov 16:00 | Demo video uploaded to Drive |
| **4 Nov 17:00** | **Working version frozen (3 hours of buffer)** |
| 4 Nov 19:00 | Submitted on the event portal, one hour before the deadline |
