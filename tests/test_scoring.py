"""Scoring behaviour against the committed model, so these run on a fresh clone with no raw data."""
from dataclasses import replace
from itertools import product

import pytest

from zerofile.scoring import FACTOR_GROUPS, Applicant, Scorer

scorer = Scorer()
TYPICAL = Applicant(vehicle_price=70_000, down_payment=15_000, employment="Self employed")


def test_points_add_up_exactly():
    result = scorer.score(TYPICAL)
    assert result.base_points + sum(result.factors.values()) == pytest.approx(result.points, abs=1e-6)


def test_every_factor_group_is_reported_once():
    result = scorer.score(TYPICAL)
    assert set(result.factors) == set(FACTOR_GROUPS)
    assert {reason["factor"] for reason in result.reasons} == set(FACTOR_GROUPS)
    sizes = [abs(reason["points"]) for reason in result.reasons]
    assert sizes == sorted(sizes, reverse=True)


def test_a_bigger_down_payment_never_raises_risk():
    risks = [scorer.score(replace(TYPICAL, down_payment=down)).risk for down in range(7_000, 63_000, 500)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(risks, risks[1:]))


def test_score_stays_in_the_advertised_range():
    tiers = ("better", "worse")
    extremes = []
    for down, dealer, branch, brand, employment in product((7_000, 60_000), tiers, tiers, tiers, ("Salaried", "Self employed")):
        extremes.append(scorer.score(Applicant(70_000, down, employment, brand=brand, dealer=dealer, branch=branch)).points)
    assert 300 <= min(extremes) and max(extremes) <= 900


def test_bands_follow_the_cutoffs():
    thresholds = scorer.meta["thresholds"]
    assert thresholds["approve"] < thresholds["high_risk"]
    assert scorer._band(thresholds["approve"]) == "Approve"
    assert scorer._band((thresholds["approve"] + thresholds["high_risk"]) / 2) == "Human review"
    assert scorer._band(thresholds["high_risk"] + 0.01) == "High risk"


@pytest.mark.parametrize("price,down", [(70_000, 70_000), (70_000, -1), (0, 0), (70_000, 80_000)])
def test_invalid_money_is_rejected(price, down):
    with pytest.raises(ValueError):
        scorer.score(Applicant(vehicle_price=price, down_payment=down))


@pytest.mark.parametrize("change", [{"employment": "Astronaut"}, {"dealer": "excellent"}])
def test_invalid_choices_are_rejected(change):
    with pytest.raises(ValueError):
        scorer.score(replace(TYPICAL, **change))


def test_out_of_range_inputs_warn_instead_of_failing():
    assert scorer.score(Applicant(vehicle_price=5_000_000, down_payment=1_000_000)).warnings
    assert not scorer.score(TYPICAL).warnings


def test_missing_employment_still_scores():
    assert 0 < scorer.score(replace(TYPICAL, employment="Not stated")).risk < 1


def test_recourse_reaches_the_approve_band():
    applicant = Applicant(vehicle_price=70_000, down_payment=7_000, employment="Self employed")
    assert scorer.score(applicant).band != "Approve"
    options = scorer.recourse(applicant)
    assert options, "A typical applicant should have a path to approval"
    for option in options:
        changed = {"down_payment": replace(applicant, down_payment=option.get("down_payment", 0)), "cheaper_vehicle": replace(applicant, vehicle_price=option.get("vehicle_price", 1))}[option["kind"]]
        assert scorer.score(changed).band == "Approve"


def test_down_payment_recourse_is_the_smallest_option():
    applicant = Applicant(vehicle_price=70_000, down_payment=7_000, employment="Self employed")
    option = next(o for o in scorer.recourse(applicant) if o["kind"] == "down_payment")
    just_below = replace(applicant, down_payment=option["down_payment"] - 500)
    assert scorer.score(just_below).band != "Approve"


def test_approved_applicants_need_no_recourse():
    applicant = Applicant(vehicle_price=70_000, down_payment=35_000, employment="Salaried", dealer="better", branch="better", brand="better")
    assert scorer.score(applicant).band == "Approve"
    assert scorer.recourse(applicant) == []


def test_recourse_never_changes_identity_or_channel():
    for option in scorer.recourse(Applicant(vehicle_price=70_000, down_payment=7_000)):
        assert not set(option) & {"dealer", "brand", "branch", "employment"}


def test_points_for_risk_inverts_the_score():
    result = scorer.score(TYPICAL)
    assert scorer.points_for_risk(result.risk) == pytest.approx(result.points, abs=1e-6)


def test_loan_range_limits_match_the_bands():
    loans = scorer.loan_range(TYPICAL)
    assert loans["approve_up_to"] <= loans["review_up_to"]
    at = lambda loan: scorer.score(replace(TYPICAL, down_payment=TYPICAL.vehicle_price - loan)).band
    assert at(loans["approve_up_to"]) == "Approve"
    assert at(loans["approve_up_to"] + 500) != "Approve"
    assert at(loans["review_up_to"]) != "High risk"
    assert at(loans["review_up_to"] + 500) == "High risk"


def test_loan_range_handles_a_very_cheap_vehicle():
    assert set(scorer.loan_range(Applicant(vehicle_price=600, down_payment=100))) == {"approve_up_to", "review_up_to"}
