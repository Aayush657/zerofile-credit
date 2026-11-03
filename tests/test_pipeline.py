"""Data, feature and metric logic, tested on small hand-built frames so no raw data is needed."""
import numpy as np
import pandas as pd
import pytest

from zerofile import evaluate as ev
from zerofile.data import BUREAU_NUMERIC, REQUIRED_COLUMNS, prepare, split
from zerofile.features import FEATURES, encode_channels, monotone_for


def raw_rows(**overrides) -> pd.DataFrame:
    base = {column: [0, 0] for column in REQUIRED_COLUMNS}
    base.update({
        "disbursed_amount": [50_000, 40_000],
        "asset_cost": [100_000, 80_000],
        "Date.of.Birth": ["01-01-68", "15-06-90"],
        "DisbursalDate": ["01-08-18", "05-10-18"],
        "Employment.Type": ["Salaried", None],
        "PERFORM_CNS.SCORE.DESCRIPTION": ["No Bureau History Available", "C-Very Low Risk"],
        "AVERAGE.ACCT.AGE": ["0yrs 0mon", "1yrs 11mon"],
        "CREDIT.HISTORY.LENGTH": ["0yrs 0mon", "2yrs 1mon"],
        "UniqueID": [1, 2],
        "Current_pincode_ID": [1441, 1502],
        "Employee_code_ID": [1998, 1998],
    })
    base.update(overrides)
    return pd.DataFrame(base)


def test_prepare_drops_personal_columns():
    frame = prepare(raw_rows())
    assert not {"Date.of.Birth", "UniqueID", "Current_pincode_ID", "Employee_code_ID", "DisbursalDate"} & set(frame.columns)


def test_prepare_fixes_two_digit_birth_years():
    frame = prepare(raw_rows())
    assert frame["age"].iloc[0] == pytest.approx(50.6, abs=0.1)  # Born 1968, not 2068.


def test_prepare_derives_fields():
    frame = prepare(raw_rows())
    assert frame["ltv_pct"].tolist() == [50.0, 50.0]
    assert frame["no_bureau"].tolist() == [True, False]
    assert frame["bureau_low_risk"].tolist() == [False, True]
    assert frame["history_months"].tolist() == [0, 25]
    assert frame["employment"].tolist() == ["Salaried", "Not stated"]
    assert np.isnan(frame["self_employed"].iloc[1])


def test_prepare_rejects_a_changed_schema():
    with pytest.raises(ValueError, match="asset_cost"):
        prepare(raw_rows().drop(columns="asset_cost"))


def test_split_is_by_time_with_no_overlap():
    dates = pd.date_range("2018-08-01", "2018-10-31")
    train, val, test = split(pd.DataFrame({"date": dates}))
    assert len(train) + len(val) + len(test) == len(dates)
    assert train["date"].max() < val["date"].min() and val["date"].max() < test["date"].min()
    assert test["date"].min() == pd.Timestamp("2018-10-01")


def test_features_exclude_bureau_age_and_region():
    assert not set(FEATURES) & (set(BUREAU_NUMERIC) | {"age", "region", "region_risk", "no_bureau", "bureau_score"})


def test_only_ltv_is_monotonic():
    assert monotone_for(["ltv_pct", "pan", "dealer_risk"]) == [1, 0, 0]


def channel_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "default": rng.integers(0, 2, rows),
        "brand": rng.integers(0, 3, rows),
        "dealer": rng.integers(0, 3, rows),
        "branch": rng.integers(0, 3, rows),
        "region": rng.integers(0, 3, rows),
    })


def test_encoding_gives_unseen_ids_the_overall_rate():
    train, later = channel_frame(500), channel_frame(10, seed=1)
    later["dealer"] = 999
    _, encoded = encode_channels(train, later)
    assert (encoded["dealer_risk"] == train["default"].mean()).all()


def test_training_encoding_is_out_of_fold():
    train = channel_frame(1000)
    train["dealer"] = np.arange(1000) // 100  # Ten dealers, so each fold holds most of each dealer.
    encoded, = encode_channels(train, smoothing=0)
    rows = train.assign(risk=encoded["dealer_risk"])
    # A row that defaulted must not lift its own dealer's rate, so defaulters see a lower mean than repayers.
    assert rows.groupby("default")["risk"].mean()[1] < rows.groupby("default")["risk"].mean()[0]


def test_pick_threshold_respects_the_target():
    p = np.linspace(0, 1, 2000)
    y = (p > 0.5).astype(int)
    cutoff = ev.pick_threshold(p, y, target=0.1, min_rows=10)
    assert y[p <= cutoff].mean() <= 0.1
    assert cutoff == pytest.approx(0.5, abs=0.08)


def test_pick_threshold_raises_when_the_target_is_unreachable():
    with pytest.raises(ValueError):
        ev.pick_threshold(np.linspace(0, 1, 100), np.ones(100, dtype=int), target=0.1, min_rows=10)


def test_policy_row_counts_people_and_crore():
    row = ev.policy_row(np.array([True, True, False]), np.array([0, 1, 1]), np.array([1e7, 1e7, 1e7]))
    assert row == {"share": pytest.approx(2 / 3), "people": 2, "default": 0.5, "crore": 2.0}


def test_rule_default_uses_lowest_values_first():
    assert ev.rule_default_at(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1]), 0.5) == 0.0


def test_classification_treats_not_approved_as_predicted_default():
    result = ev.classification_at(np.array([True, True, False, False]), np.array([0, 1, 1, 0]))
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["accuracy"] == 0.5


def test_group_audit_flags_the_four_fifths_rule():
    groups = np.array(["a"] * 100 + ["b"] * 100 + ["tiny"] * 5)
    approved = np.array([True] * 80 + [False] * 20 + [True] * 40 + [False] * 60 + [True] * 5)
    audit = {row["group"]: row for row in ev.group_audit(groups, approved, np.zeros(205, dtype=int), min_rows=50)}
    assert set(audit) == {"a", "b"}  # Small groups are skipped.
    assert audit["a"]["passes_four_fifths"] and not audit["b"]["passes_four_fifths"]
    assert audit["b"]["ratio"] == pytest.approx(0.5)


def test_regional_floor_lifts_low_regions_with_their_safest_applicants():
    regions = np.array(["a"] * 100 + ["b"] * 100)
    p = np.concatenate([np.linspace(0, 0.1, 100), np.linspace(0.5, 1.0, 100)])
    approved = p <= 0.1  # Region a all approved, b none, so overall is 50% and the floor is 40%.
    result = ev.regional_floor(p, np.zeros(200, dtype=int), np.ones(200), regions, approved, min_rows=50)
    assert result["regions_below_before"] == 1 and result["regions_below_after"] == 0
    assert result["people_added"] == 40 and result["people"] == 140
