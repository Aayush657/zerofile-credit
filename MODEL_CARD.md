# Model card: ZeroFile

ZeroFile has two models, kept apart on purpose. **Only the vehicle-loan model has been tested against real repayment outcomes.**

## Purpose

Estimate the risk of default for people who have no credit bureau history, so that they can be considered at all.

**Intended use:** a research prototype and decision aid that sends borderline cases to a person.

**Not intended for:** declining anyone automatically, setting interest rates, or use in production without local validation and legal review.

## Vehicle-loan model

| | |
|---|---|
| Type | LightGBM binary classifier, 152 trees, Platt calibration |
| Target | Default on the first instalment |
| Constraint | Risk can never fall as loan-to-value rises |
| Inputs (6) | Loan-to-value, vehicle price, employment type, and the historic repayment records of the vehicle brand, dealer and branch |
| Output | Risk estimate, a 300 to 900 score, a band, SHAP reasons, a suggested loan range, and recourse |

### Inputs deliberately left out

| Input | Why | Cost in AUC on no-file applicants |
|---|---|---|
| Bureau data | The whole point. These applicants have none | None for them |
| ID-document flags | Penalised people for which identity document they shared | About 0.001 |
| Age | Should not drive a credit decision | About 0.002 together with region |
| Region | Direct location signal | About 0.002 together with age |
| Pincode | Neighbourhood-level location signal, the classic redlining route | Dropped at load |
| Employee code | Records who logged the loan, not anything about the applicant | Dropped at load |

### Data

[L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction): 233,154 Indian vehicle loans disbursed from August to October 2018, of which 116,950 have no bureau history. The licence is unclear, so raw data is not redistributed.

| Split | Dates | Loans | Default rate |
|---|---|---|---|
| Train | 1 Aug to 23 Sep 2018 | 111,622 | 20.4% |
| Validation (early stopping, calibration, cutoffs) | 24 to 30 Sep 2018 | 23,168 | 20.2% |
| Test | 2 to 31 Oct 2018 | 98,364 | 23.5% |

### Performance on no-file applicants in the test month (49,748)

| Metric | Value |
|---|---|
| AUC | 0.631 (95% CI 0.626 to 0.636) |
| Logistic regression on the same inputs | 0.626 |
| KS statistic | 0.189 |
| Brier score | 0.178 (base rate alone: 0.183) |
| Expected calibration error | 4.1% |
| Approved at the fixed cutoff | 42.8% (21,288 people, ₹107 crore) |
| Default rate among approved | 16.1% (not approved: 30.0%) |
| Bureau low-risk benchmark | 19.0% |

A scan of 32 LightGBM settings moved the AUC only between 0.629 and 0.631, so the settings are not what limits it.

### Fairness

Audited with the four-fifths rule on approval rate, plus default rate and approval among those who repaid. Age band and employment type pass in full. **Region fails: 13 of 18 regions fall under four-fifths of the best region.** Examples:

- Region 14: 5.7% approved, with a 32.0% default rate. The direction is right, but the gap is far wider than the risk gap.
- Region 11: 29.5% approved, despite a below-average 20.2% default rate.
- Region 12: 65.9% approved, and 30.1% of those default. The model is missing something there.

A regional approval floor adds 2,143 people at a cost of 0.7 points of default (16.8%). It raises the lowest region from 8% to 49% of the highest. The gap narrows but does not close. The floor was computed on October after the fact and has not been tested on a later period.

Gender and income are not in this dataset, so they could not be audited. Employment type (salaried or self-employed) is the only occupation signal.

## Worker model

**Everything in this section comes from synthetic data. It shows how such a score would work, and proves nothing about real accuracy.**

| | |
|---|---|
| Type | LightGBM binary classifier, 245 trees, Platt calibration |
| Target | The dataset's `defaulted` flag |
| Constraints | Better payment habits can never raise risk, and a heavier instalment or wilder income can never lower it |
| Inputs (7) | Utility bills and rent paid on time, mobile recharges a month, UPI payments a month and months active, income swing, and instalment against income |
| Policy rule | Suggested loans are capped so the instalment stays under 40% of income. This is a guardrail and was not learned |

### Data

[Alternate Credit Dataset](https://www.kaggle.com/datasets/shreyyans/alternate-credit-dataset) (MIT licence): 10,000 synthetic Indian borrowers, 5,000 gig, 3,000 rural and 2,000 migrant. **`defaulted` is simply `default_probability` above a fixed cutoff**, so `default_probability` is the label in disguise and is left out of training. A stratified 60/20/20 split is used, because the data has no dates.

### Results on the synthetic test set (2,000)

| Metric | Value |
|---|---|
| AUC | 0.933 (95% CI 0.920 to 0.946) |
| AUC across 5 random splits | 0.922 to 0.934 |
| Logistic regression on the same inputs | 0.887 |
| Brier score | 0.068 (base rate alone: 0.128) |
| Expected calibration error | 1.1% |

| Band | Rule, fixed in advance | Share | Default rate |
|---|---|---|---|
| Approve | Risk at or below 15%, the average borrower | 71.7% | 2.2% |
| Human review | In between | 16.9% | 29.0% |
| High risk | Risk above 50%, more likely to default than not | 11.5% | 74.7% |

### Inputs deliberately left out

Income level, recharge amount and UPI transaction size are wealth stand-ins, and dropping them cost nothing. Worker type, state, age, household size, survey answers, e-commerce, job tenure, income sources and mobile-number age added almost nothing.

### Fairness

Each group is tested on being approved outright, and on not being marked high risk, which is everyone who reaches a yes or a person.

- **Rural workers:** 51% approved outright (fails), 78% reach a yes or a person (passes).
- **No UPI history:** 35% approved outright (fails), 67% reach a yes or a person against 93% of UPI users (fails). The data has them defaulting at 41% against 10%. That may be the dataset author's assumption, so every no-UPI result is flagged for a person.
- **States:** 10 of 11 pass on reaching a yes or a person. Madhya Pradesh does not.
- **Smaller loans:** these would lift rural approval from 51% to 69%, but 30% of those people defaulted on the larger loan they asked for. This data cannot say whether a smaller one would have helped, so the advice is untested.

Gender is not in this dataset, so it could not be audited.

## Limits

1. **No real alternative data.** No public dataset I could find links utility or mobile records to repayment. A real deployment would need a consented pilot before any of the worker model's numbers mean anything.
2. **Synthetic labels.** The worker model's accuracy mostly shows it recovering the generator's rules. Its risk gaps by region and UPI access are the dataset author's assumptions.
3. **Selection bias.** Every real row is a loan the lender already funded. Results may not carry over to applicants it rejected, who are the people this project most wants to reach.
4. **Thin-file is not unbanked.** The real borrowers applied for vehicle finance through a dealer. Most probably have bank accounts, and the data does not mark rural applicants or specific gig jobs.
5. **Age of the data.** One lender, three months, 2018. It predates much of India's digital-credit growth.
6. **Modest real signal.** AUC near 0.63 separates groups. It does not predict individuals.
7. **Calibration drift.** Actual defaults reached 27.5% in the last week of October against 20.1% predicted. Ranking held up better than absolute risk, and the bands report what actually happened for that reason.
8. **Test not fully blind.** October was inspected while choosing features. Those choices only removed inputs, which should not inflate results, but no untouched labelled period exists. The dataset's November file has no labels.
9. **Not causal.** A bigger down payment or better bill record lowers the estimate. It does not prove the applicant will repay, and a higher down payment may simply select people with more savings.
10. **Gaming.** A live system would need checks that stated figures are real and that dealer records cannot be manipulated.

## Reproducing

Run `python -m zerofile.data`, then `python -m zerofile.train`, `python -m zerofile.worker` and `python -m zerofile.tune`. These rewrite everything in `artifacts/`, with fixed seeds.
