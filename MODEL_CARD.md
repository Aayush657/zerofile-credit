# Model card: ZeroFile

## Purpose

Estimate the risk of default on the first instalment for vehicle-loan applicants who have no credit bureau history, so that they can be considered at all.

**Intended use:** a research prototype and decision aid that sends borderline cases to a person.

**Not intended for:** declining anyone automatically, setting interest rates, loans other than two-wheeler-style vehicle finance, or use in production without local validation and legal review.

## Model

| | |
|---|---|
| Type | LightGBM binary classifier, 152 trees, Platt calibration |
| Target | Default on the first instalment |
| Constraint | Risk can never fall as loan-to-value rises |
| Inputs (6) | Loan-to-value, vehicle price, employment type, and the historic repayment records of the vehicle brand, dealer and branch |
| Output | Risk estimate, a 300 to 900 score, a band, reasons, and recourse |

## Inputs deliberately left out

| Input | Why | Cost in AUC on no-file applicants |
|---|---|---|
| Bureau data | The whole point. These applicants have none | None for them |
| ID-document flags | Penalised people for which identity document they shared | About 0.001 |
| Age | Should not drive a credit decision | About 0.002 together with region |
| Region | Direct location signal | About 0.002 together with age |
| Pincode | Neighbourhood-level location signal, the classic redlining route | Dropped at load |
| Employee code | Records who logged the loan, not anything about the applicant | Dropped at load |

## Data

[L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction): 233,154 Indian vehicle loans disbursed from August to October 2018, of which 116,950 have no bureau history. The licence is unclear, so raw data is not redistributed.

| Split | Dates | Loans | Default rate |
|---|---|---|---|
| Train | 1 Aug to 23 Sep 2018 | 111,622 | 20.4% |
| Validation (early stopping, calibration, cutoffs) | 24 to 30 Sep 2018 | 23,168 | 20.2% |
| Test | 2 to 31 Oct 2018 | 98,364 | 23.5% |

## Performance on no-file applicants in the test month (49,748)

| Metric | Value |
|---|---|
| AUC | 0.631 (95% CI 0.626 to 0.636) |
| KS statistic | 0.189 |
| Brier score | 0.178 (base rate alone: 0.183) |
| Expected calibration error | 4.1% |
| Approved at the fixed cutoff | 42.8% (21,288 people, ₹107 crore) |
| Default rate among approved | 16.1% (not approved: 30.0%) |
| Bureau low-risk benchmark | 19.0% |

## Fairness

Audited with the four-fifths rule on approval rate, plus default rate and approval among those who repaid. Age band and employment type pass in full. **Region fails: 13 of 18 regions fall under four-fifths of the best region.** Examples:

- Region 14: 5.7% approved, with a 32.0% default rate. The direction is right, but the gap is far wider than the risk gap.
- Region 11: 29.5% approved, despite a below-average 20.2% default rate.
- Region 12: 65.9% approved, and 30.1% of those default. The model is missing something there.

A regional approval floor adds 2,143 people at a cost of 0.7 points of default (16.8%). It raises the lowest region from 8% to 49% of the highest. The gap narrows but does not close. The floor was computed on October after the fact and has not been tested on a later period.

Gender and income are not in the dataset, so they could not be audited.

## Limits

1. **Selection bias.** Every row is a loan the lender already funded. Results may not carry over to applicants it rejected, who are the people this project most wants to reach.
2. **Thin-file is not unbanked.** These borrowers applied for vehicle finance through a dealer. Most probably have bank accounts.
3. **Age of the data.** One lender, three months, 2018. It predates much of India's digital-credit growth.
4. **Modest signal.** AUC near 0.63 separates groups. It does not predict individuals.
5. **Calibration drift.** Actual defaults reached 27.5% in the last week of October against 20.1% predicted. Ranking held up better than absolute risk, and the bands report what actually happened for that reason.
6. **Test not fully blind.** October was inspected while choosing features. Those choices only removed inputs, which should not inflate results, but no untouched labelled period exists. The dataset's November file has no labels.
7. **Not causal.** A bigger down payment lowers the estimate. It does not prove the applicant will repay, and a higher down payment may simply select people with more savings.
8. **Gaming.** A live system would need checks that a stated down payment is real and that dealer records cannot be manipulated.

## Reproducing

`python -m zerofile.data`, then `python -m zerofile.train`. This rewrites everything in `artifacts/`, with a fixed seed.
