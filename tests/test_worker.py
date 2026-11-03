"""The alternative-data model: its data handling, its guardrails, and its behaviour against the committed model."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from zerofile.scoring import AFFORDABILITY_CAP, WORKER_FACTOR_GROUPS, Worker, WorkerScorer, net_value
from zerofile.worker import (
    DEMOGRAPHICS,
    OTHER_INPUTS,
    REQUIRED_COLUMNS,
    WEALTH_STAND_INS,
    WORKER_FEATURES,
    monotone_for,
    prepare,
    split,
)

scorer = WorkerScorer()
TYPICAL = Worker(0.65, 1.75, 24_000, 0.4, 120_000, 36, rent_on_time=0.6, upi_transactions=45, upi_months=15)
NO_UPI = Worker(0.45, 1.5, 13_000, 0.35, 60_000, 24)


def raw_rows(rows: int = 4) -> pd.DataFrame:
    base = {column: np.arange(1, rows + 1, dtype=float) * 100 for column in REQUIRED_COLUMNS}
    base.update(
        {
            "borrower_type": ["gig", "rural"] * (rows // 2),
            "state": ["Bihar", "Delhi"] * (rows // 2),
            "employment_type": ["seasonal", "daily-wage"] * (rows // 2),
            "defaulted": [0, 1] * (rows // 2),
            "utility_bills_paid": [6.0] * rows,
            "utility_bills_total": [12.0] * rows,
            "rent_paid_on_time_months": [np.nan, 5.0] * (rows // 2),
            "total_rental_months": [np.nan, 10.0] * (rows // 2),
            "upi_transactions_per_month": [np.nan, 20.0] * (rows // 2),
            "same_number_since_year": [2020.0] * rows,
            "loan_amount_requested": [120_000.0] * rows,
            "loan_tenure_months": [12.0] * rows,
        }
    )
    for month in range(1, 7):
        base[f"income_month_{month}"] = [10_000.0] * rows
    return pd.DataFrame(base).assign(borrower_id="B1", default_probability=0.5)


def test_prepare_leaves_the_label_in_disguise_behind():
    frame = prepare(raw_rows())
    assert "default_probability" not in frame.columns and "borrower_id" not in frame.columns


def test_prepare_derives_the_inputs():
    frame = prepare(raw_rows())
    assert frame["bills_on_time"].iloc[0] == 0.5
    assert np.isnan(frame["rent_on_time"].iloc[0]) and frame["rent_on_time"].iloc[1] == 0.5
    assert frame["emi_to_income"].iloc[0] == pytest.approx(1.0)  # 120,000 over 12 months against 10,000 a month.
    assert frame["income_swing"].iloc[0] == 0
    assert frame["uses_upi"].tolist()[:2] == ["No UPI history", "Uses UPI"]


def test_prepare_rejects_a_changed_schema():
    with pytest.raises(ValueError, match="utility_bills_paid"):
        prepare(raw_rows().drop(columns="utility_bills_paid"))


def test_split_has_no_overlap_and_keeps_proportions():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"borrower_type": rng.choice(["gig", "rural"], 1000), "default": rng.integers(0, 2, 1000)})
    train, val, test = split(frame)
    assert len(train) == 600 and len(val) == 200 and len(test) == 200
    assert not (
        set(train.index) & set(val.index) or set(train.index) & set(test.index) or set(val.index) & set(test.index)
    )
    assert abs(test["default"].mean() - frame["default"].mean()) < 0.02


def test_inputs_exclude_the_label_identity_and_wealth():
    assert not set(WORKER_FEATURES) & set(
        WEALTH_STAND_INS
        + OTHER_INPUTS
        + DEMOGRAPHICS
        + ["default_probability", "borrower_type", "state", "income_mean"]
    )
    assert scorer.features == WORKER_FEATURES


def test_better_habits_can_only_lower_risk_and_heavier_loans_can_only_raise_it():
    constraints = dict(zip(WORKER_FEATURES, monotone_for(WORKER_FEATURES), strict=False))
    assert constraints["bills_on_time"] == -1 and constraints["upi_transactions"] == -1
    assert constraints["emi_to_income"] == 1 and constraints["income_swing"] == 1


def test_points_add_up_exactly():
    result = scorer.score(TYPICAL)
    assert result.base_points + sum(result.factors.values()) == pytest.approx(result.points, abs=1e-6)
    assert set(result.factors) == set(WORKER_FACTOR_GROUPS)


def test_score_stays_in_the_advertised_range():
    scores = []
    for bills, recharges, swing, loan, upi, rent, (income, tenure) in product(
        (0.0, 1.0),
        (0.5, 3.75),
        (0.0, 0.8),
        (5_000, 2_000_000),
        ((None, None), (120, 36), (0, 0)),
        (None, 0.0, 1.0),
        ((1_000, 6), (60_000, 60)),
    ):
        scores.append(
            scorer.score(
                Worker(
                    bills,
                    recharges,
                    income,
                    swing,
                    loan,
                    tenure,
                    rent_on_time=rent,
                    upi_transactions=upi[0],
                    upi_months=upi[1],
                )
            ).points
        )
    assert 300 <= min(scores) and max(scores) <= 900


def test_points_for_risk_inverts_the_score():
    result = scorer.score(TYPICAL)
    assert scorer.points_for_risk(result.risk) == pytest.approx(result.points, abs=1e-6)


@pytest.mark.parametrize("field", ["bills_on_time", "rent_on_time"])
def test_paying_more_on_time_never_raises_risk(field):
    risks = [scorer.score(replace(TYPICAL, **{field: share / 100})).risk for share in range(0, 101, 5)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(risks, risks[1:], strict=False))


def test_a_bigger_loan_never_lowers_risk():
    risks = [scorer.score(replace(TYPICAL, loan_amount=loan)).risk for loan in range(5_000, 500_001, 5_000)]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(risks, risks[1:], strict=False))


def test_bands_are_cutoffs_on_a_persons_own_risk():
    thresholds = scorer.meta["thresholds"]
    assert thresholds["approve"] == pytest.approx(0.15, abs=0.001) and thresholds["high_risk"] == 0.5
    assert scorer._band(0.1) == "Approve" and scorer._band(0.3) == "Human review" and scorer._band(0.6) == "High risk"


def test_no_upi_history_is_flagged_for_a_person():
    assert any("no UPI history" in note for note in scorer.score(NO_UPI).warnings)
    assert not any("UPI" in note for note in scorer.score(TYPICAL).warnings)


def test_an_unaffordable_loan_is_flagged_whatever_the_score():
    loan = int(TYPICAL.monthly_income * TYPICAL.tenure_months * (AFFORDABILITY_CAP + 0.1))
    assert any("affordability" in note for note in scorer.score(replace(TYPICAL, loan_amount=loan)).warnings)
    assert not scorer.score(TYPICAL).warnings


def test_loan_range_never_passes_the_affordability_cap():
    loans = scorer.loan_range(TYPICAL)
    cap = AFFORDABILITY_CAP * TYPICAL.monthly_income * TYPICAL.tenure_months
    assert loans["approve_up_to"] <= loans["review_up_to"] <= cap


def test_loan_range_limits_match_the_bands():
    worker = replace(NO_UPI, bills_on_time=0.6)
    loans = scorer.loan_range(worker)
    assert scorer.score(replace(worker, loan_amount=loans["approve_up_to"])).band == "Approve"
    assert scorer.score(replace(worker, loan_amount=loans["approve_up_to"] + 1_000)).band != "Approve"


def test_loan_range_handles_income_too_small_for_any_loan():
    assert scorer.loan_range(replace(TYPICAL, monthly_income=100, tenure_months=6)) == {
        "approve_up_to": None,
        "review_up_to": None,
    }


def test_recourse_reaches_the_approve_band():
    assert scorer.score(NO_UPI).band != "Approve"
    options = scorer.recourse(NO_UPI)
    assert options, "This worker should have a path to approval"
    for option in options:
        change = (
            {"loan_amount": option["loan_amount"]}
            if option["kind"] == "smaller_loan"
            else {"bills_on_time": option["bills_on_time"]}
        )
        assert scorer.score(replace(NO_UPI, **change)).band == "Approve"


def test_approved_workers_need_no_recourse():
    assert scorer.score(TYPICAL).band == "Approve" and scorer.recourse(TYPICAL) == []


@pytest.mark.parametrize(
    "change",
    [
        {"monthly_income": 0},
        {"loan_amount": 0},
        {"tenure_months": 0},
        {"bills_on_time": 1.5},
        {"rent_on_time": -0.1},
        {"income_swing": 2.0},
        {"recharges_per_month": -1},
        {"upi_months": None},
    ],
)
def test_invalid_inputs_are_rejected(change):
    with pytest.raises(ValueError):
        scorer.score(replace(TYPICAL, **change))


def test_net_value_breaks_even_where_the_economics_say():
    margin, loss = 0.12, 0.5
    assert net_value(100, margin / (margin + loss), margin, loss) == pytest.approx(0)
    assert net_value(100, 0.0, margin, loss) == pytest.approx(12)
    assert net_value(100, 1.0, margin, loss) == pytest.approx(-50)
