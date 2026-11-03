"""Alternative-data model for gig, rural and migrant workers. `python -m zerofile.worker` rewrites artifacts/worker_*.

The training data is synthetic: `defaulted` is simply `default_probability` above a fixed cutoff. High accuracy here
shows the model recovering the generator's rules, not real-world skill, and must be labelled that way everywhere.
"""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zerofile import evaluate as ev
from zerofile.data import WORKER_DATA_FILE, data_dir
from zerofile.scoring import ARTIFACTS, WORKER_FACTOR_GROUPS, WORKER_SCALE, Worker, WorkerScorer
from zerofile.train import calibrate, calibrated_risk

BILLS = ["bills_on_time", "rent_on_time"]
MOBILE = ["recharges_per_month"]
UPI = ["upi_transactions", "upi_months"]
INCOME = ["income_swing"]
LOAN = ["emi_to_income"]
WORKER_FEATURES = BILLS + MOBILE + UPI + INCOME + LOAN
# Measured and rejected. Wealth stand-ins cost nothing to drop, and the rest add no signal while inviting bias.
WEALTH_STAND_INS = ["upi_average_amount", "recharge_amount", "income_mean"]
LOW_VALUE = ["sim_years", "income_sources", "months_in_work"]
OTHER_INPUTS = ["ecomm_orders_per_month", "ecomm_return_rate", "prepaid_orders_ratio"] + [
    f"survey_q{i}" for i in range(1, 9)
]
DEMOGRAPHICS = ["age", "household_size", "borrower_type_code", "state_code", "employment_code"]

LOWER_RISK_WHEN_HIGHER = {
    "bills_on_time",
    "rent_on_time",
    "recharges_per_month",
    "sim_years",
    "upi_transactions",
    "upi_months",
    "income_sources",
    "months_in_work",
    "upi_average_amount",
    "recharge_amount",
    "income_mean",
}
HIGHER_RISK_WHEN_HIGHER = {"income_swing", "emi_to_income"}
HIGH_RISK_AT = 0.5  # More likely to default than not.
PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    seed=7,
    deterministic=True,
    force_row_wise=True,
    n_jobs=8,
    verbose=-1,
)
REFERENCE_YEAR = 2025  # The dataset's mobile-number years run to 2024.
VARIANTS = {
    "Utility bills and rent only": BILLS,
    "+ mobile recharges": BILLS + MOBILE,
    "+ UPI use": BILLS + MOBILE + UPI,
    "+ income steadiness": BILLS + MOBILE + UPI + INCOME,
    "ZeroFile worker model: + loan burden": WORKER_FEATURES,
    "+ wealth stand-ins (rejected)": WORKER_FEATURES + WEALTH_STAND_INS,
    "+ job tenure, income sources, number age (rejected)": WORKER_FEATURES + LOW_VALUE,
    "+ survey and e-commerce (rejected)": WORKER_FEATURES + OTHER_INPUTS,
    "+ demographics (rejected)": WORKER_FEATURES + DEMOGRAPHICS,
    "Demographics alone": DEMOGRAPHICS,
}
REQUIRED_COLUMNS = {
    "borrower_type",
    "state",
    "employment_type",
    "age",
    "household_size",
    "utility_bills_paid",
    "utility_bills_total",
    "rent_paid_on_time_months",
    "total_rental_months",
    "recharge_frequency_per_month",
    "avg_monthly_recharge_amount",
    "same_number_since_year",
    "upi_transactions_per_month",
    "upi_months_active",
    "upi_avg_transaction_amount",
    "num_income_sources",
    "months_at_current_job",
    "loan_amount_requested",
    "loan_tenure_months",
    "defaulted",
    *[f"income_month_{i}" for i in range(1, 7)],
    *OTHER_INPUTS,
}


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's inputs. `default_probability` is deliberately left behind, because it is the label in disguise."""
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    income = raw[[f"income_month_{i}" for i in range(1, 7)]]
    income_mean = income.mean(axis=1)
    frame = pd.DataFrame(
        {
            "default": raw["defaulted"],
            "loan": raw["loan_amount_requested"],
            "bills_on_time": raw["utility_bills_paid"] / raw["utility_bills_total"],
            "rent_on_time": raw["rent_paid_on_time_months"] / raw["total_rental_months"],
            "recharges_per_month": raw["recharge_frequency_per_month"],
            "upi_transactions": raw["upi_transactions_per_month"],
            "upi_months": raw["upi_months_active"],
            "income_swing": income.std(axis=1) / income_mean,
            "emi_to_income": (raw["loan_amount_requested"] / raw["loan_tenure_months"]) / income_mean,
            "sim_years": REFERENCE_YEAR - raw["same_number_since_year"],
            "income_sources": raw["num_income_sources"],
            "months_in_work": raw["months_at_current_job"],
            "income_mean": income_mean,
            "upi_average_amount": raw["upi_avg_transaction_amount"],
            "recharge_amount": raw["avg_monthly_recharge_amount"],
            "age": raw["age"],
            "household_size": raw["household_size"],
            "borrower_type": raw["borrower_type"],
            "state": raw["state"],
            "employment": raw["employment_type"],
            "borrower_type_code": raw["borrower_type"].astype("category").cat.codes,
            "state_code": raw["state"].astype("category").cat.codes,
            "employment_code": raw["employment_type"].astype("category").cat.codes,
            "uses_upi": np.where(raw["upi_transactions_per_month"].notna(), "Uses UPI", "No UPI history"),
        }
    )
    return pd.concat([frame, raw[OTHER_INPUTS]], axis=1)


def split(frame: pd.DataFrame, seed: int = 7) -> tuple:
    """Random 60/20/20, keeping borrower types and outcomes in proportion. This data has no dates to split on."""
    strata = frame["borrower_type"] + frame["default"].astype(str)
    rest, test = train_test_split(frame, test_size=0.2, random_state=seed, stratify=strata)
    train, val = train_test_split(rest, test_size=0.25, random_state=seed, stratify=strata.loc[rest.index])
    return train, val, test


def monotone_for(features: list[str]) -> list[int]:
    """Better habits can never raise risk, and a heavier instalment or wilder income can never lower it."""
    return [-1 if name in LOWER_RISK_WHEN_HIGHER else 1 if name in HIGHER_RISK_WHEN_HIGHER else 0 for name in features]


def fit(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> lgb.Booster:
    return lgb.train(
        {**PARAMS, "monotone_constraints": monotone_for(features)},
        lgb.Dataset(train[features], train["default"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(val[features], val["default"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def smaller_loan_check(booster: lgb.Booster, platt: dict, test: pd.DataFrame, approve_at: float) -> dict:
    """How many people outside the approve band would reach it with a smaller loan, and how did they do on the loan they asked for?"""

    def risk(rows):
        return calibrated_risk(booster, WORKER_FEATURES, platt, rows)

    approved = risk(test) <= approve_at
    outside = test[~approved]
    rescued = np.zeros(len(outside), dtype=bool)
    for keep in np.arange(0.95, 0.04, -0.05):
        rescued |= risk(outside.assign(emi_to_income=outside["emi_to_income"] * keep)) <= approve_at
    by_type = []
    for name, part in test.assign(approved=approved).groupby("borrower_type"):
        extra = int(rescued[(outside["borrower_type"] == name).to_numpy()].sum())
        by_type.append(
            {
                "group": name,
                "n": int(len(part)),
                "approved": float(part["approved"].mean()),
                "with_smaller_loan": float((part["approved"].sum() + extra) / len(part)),
            }
        )
    return {
        "by_type": by_type,
        "rescued": int(rescued.sum()),
        "rescued_default_on_loan_asked": float(outside["default"].to_numpy()[rescued].mean()),
    }


def main():
    started = time.time()
    path = data_dir() / WORKER_DATA_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m zerofile.data")
    frame = prepare(pd.read_csv(path))
    train, val, test = split(frame)

    booster = fit(train, val, WORKER_FEATURES)
    platt = calibrate(booster, WORKER_FEATURES, val)
    # Cutoffs on each person's own risk, fixed in advance and easy to say out loud: no riskier than the average borrower, and more likely to default than not.
    thresholds = {"approve": float(train["default"].mean()), "high_risk": HIGH_RISK_AT}

    ARTIFACTS.mkdir(exist_ok=True)
    booster.save_model(str(ARTIFACTS / "worker_model.txt"))
    meta = {
        "features": WORKER_FEATURES,
        "platt": platt,
        "thresholds": thresholds,
        "emi_to_income_max": float(frame["emi_to_income"].quantile(0.99)),
    }
    (ARTIFACTS / "worker_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    y, p = test["default"].to_numpy(), calibrated_risk(booster, WORKER_FEATURES, platt, test)
    loans = test["loan"].to_numpy()
    approved, high = p <= thresholds["approve"], p > thresholds["high_risk"]
    review = ~approved & ~high
    stability = []
    for seed in range(5):
        tr, va, te = split(frame, seed)
        stability.append(
            float(ev.roc_auc_score(te["default"], fit(tr, va, WORKER_FEATURES).predict(te[WORKER_FEATURES])))
        )
    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        train[WORKER_FEATURES], train["default"]
    )
    variants = []
    for name, features in VARIANTS.items():
        model = fit(train, val, features)
        audit = ev.group_audit(
            test["borrower_type"].to_numpy(), model.predict(test[features]) <= HIGH_RISK_AT, y, min_rows=50
        )
        variants.append(
            {
                "name": name,
                "auc": float(ev.roc_auc_score(y, model.predict(test[features]))),
                "worst_type_ratio": min(row["ratio"] for row in audit),
            }
        )
    sample = test[WORKER_FEATURES]
    contributions = booster.predict(sample, pred_contrib=True)[:, :-1]
    importance = [
        {
            "factor": factor,
            "points": float(
                np.abs(contributions[:, [WORKER_FEATURES.index(m) for m in members]].sum(axis=1)).mean()
                * WORKER_SCALE["per_log_odds"]
            ),
        }
        for factor, members in WORKER_FACTOR_GROUPS.items()
    ]
    groups = {"Worker type": "borrower_type", "Employment": "employment", "State": "state", "UPI access": "uses_upi"}

    report = {
        "rows": int(len(frame)),
        "splits": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "default_rate": float(frame["default"].mean()),
        "borrower_types": {
            name: {"n": int(len(part)), "default": float(part["default"].mean())}
            for name, part in frame.groupby("borrower_type")
        },
        "auc": ev.bootstrap_auc(y, p),
        "auc_by_split": stability,
        "auc_logistic": float(ev.roc_auc_score(y, logistic.predict_proba(test[WORKER_FEATURES])[:, 1])),
        "ks": ev.ks_statistic(y, p),
        "brier": ev.brier(y, p),
        "brier_base_rate": ev.brier(y, np.full(len(y), y.mean())),
        "ece": ev.expected_calibration_error(y, p),
        "headline": {
            **ev.policy_row(approved, y, loans),
            "not_approved_default": float(y[~approved].mean()),
            "classification": ev.classification_at(approved, y),
        },
        "bands": [
            {"band": "Approve", **ev.policy_row(approved, y, loans)},
            {"band": "Human review", **ev.policy_row(review, y, loans)},
            {"band": "High risk", **ev.policy_row(high, y, loans)},
        ],
        "variants": variants,
        # Two tests of each group: approved outright, and not marked high risk, which is everyone who reaches a yes or a person.
        "fairness": {
            name: ev.group_audit(test[column].to_numpy(), approved, y, min_rows=50) for name, column in groups.items()
        },
        "fairness_not_high_risk": {
            name: ev.group_audit(test[column].to_numpy(), ~high, y, min_rows=50) for name, column in groups.items()
        },
        "smaller_loan": smaller_loan_check(booster, platt, test, thresholds["approve"]),
        "importance": sorted(importance, key=lambda row: -row["points"]),
        "model": {"trees": int(booster.best_iteration), "features": len(WORKER_FEATURES)},
    }

    scorer = WorkerScorer()
    worker = Worker(0.8, 2.0, 25_000, 0.3, 150_000, 36, rent_on_time=0.8, upi_transactions=50, upi_months=20)
    timings = []
    for _ in range(200):
        tick = time.perf_counter()
        scorer.score(worker)
        timings.append((time.perf_counter() - tick) * 1000)
    report["latency_ms_p50"] = float(np.median(timings))
    (ARTIFACTS / "worker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Finished in {time.time() - started:.0f}s. Trees: {booster.best_iteration}. Thresholds {thresholds}.")
    print(
        "AUC", report["auc"], "by split", [round(v, 4) for v in stability], "logistic", round(report["auc_logistic"], 4)
    )
    print(
        {k: round(report[k], 4) for k in ("ks", "brier", "brier_base_rate", "ece")},
        "latency",
        round(report["latency_ms_p50"], 2),
    )
    print(pd.DataFrame(report["bands"]).round(3).to_string())
    print(pd.DataFrame(variants).round(4).to_string())
    for name, rows in report["fairness"].items():
        print(name, "| approved outright")
        print(pd.DataFrame(rows).round(3).to_string())
        print(name, "| not high risk")
        print(pd.DataFrame(report["fairness_not_high_risk"][name]).round(3).to_string())
    print("Smaller loan", report["smaller_loan"])
    print(pd.DataFrame(report["importance"]).round(1).to_string())


if __name__ == "__main__":
    main()
