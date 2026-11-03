"""End-to-end training run. `python -m zerofile.train` rewrites everything in artifacts/."""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zerofile import evaluate as ev
from zerofile.altdata import SCENARIOS, simulate
from zerofile.data import BUREAU_FEATURES, load, split
from zerofile.features import ALT_SIGNALS, CHANNEL, FEATURES, ID_DOCUMENTS, LOAN, PROFILE, encode_channels, monotone_for
from zerofile.scoring import ARTIFACTS, FACTOR_GROUPS, Applicant, Scorer, points_from_margin

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
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
VARIANTS = {
    "Loan terms only": LOAN,
    "+ employment type": LOAN + PROFILE,
    "ZeroFile: + brand, dealer, branch": FEATURES,
    "ZeroFile + ID documents (rejected)": FEATURES + ID_DOCUMENTS,
    "ZeroFile + age and region (rejected)": FEATURES + ["age", "region_risk"],
    "ZeroFile + bureau data (comparison)": FEATURES + BUREAU_FEATURES,
}
HIGH_RISK_SHARE = 0.2  # The riskiest fifth of validation no-bureau applicants go to the high-risk band.


def fit(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> lgb.Booster:
    params = {**PARAMS, "monotone_constraints": monotone_for(features)}
    return lgb.train(
        params,
        lgb.Dataset(train[features], train["default"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(val[features], val["default"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def calibrate(booster: lgb.Booster, features: list[str], val: pd.DataFrame) -> dict:
    platt = LogisticRegression(C=1e6).fit(booster.predict(val[features], raw_score=True).reshape(-1, 1), val["default"])
    return {"slope": float(platt.coef_[0, 0]), "intercept": float(platt.intercept_[0])}


def calibrated_risk(booster: lgb.Booster, features: list[str], platt: dict, rows: pd.DataFrame) -> np.ndarray:
    return 1 / (1 + np.exp(-(platt["slope"] * booster.predict(rows[features], raw_score=True) + platt["intercept"])))


def fix_cutoffs(booster: lgb.Booster, features: list[str], platt: dict, val: pd.DataFrame) -> dict:
    """Everything policy-related is fixed on validation, before October is scored."""
    val_nb = val[val["no_bureau"]]
    p = calibrated_risk(booster, features, platt, val_nb)
    target = float(val.loc[val["bureau_low_risk"], "default"].mean())
    return {
        "approve": ev.pick_threshold(p, val_nb["default"].to_numpy(), target),
        "high_risk": float(np.quantile(p, 1 - HIGH_RISK_SHARE)),
    }


def alt_scenarios(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """What would utility and mobile data be worth IF it were this strong? A simulation, never a measured result."""
    features = FEATURES + ALT_SIGNALS
    rows = []
    for name, strength in SCENARIOS.items():
        tr, va, te = (
            frame.assign(**simulate(frame["default"].to_numpy(), strength, seed))
            for frame, seed in ((train, 1), (val, 2), (test, 3))
        )
        booster = fit(tr, va, features)
        platt = calibrate(booster, features, va)
        cutoffs = fix_cutoffs(booster, features, platt, va)
        te_nb = te[te["no_bureau"]]
        y, p = te_nb["default"].to_numpy(), calibrated_risk(booster, features, platt, te_nb)
        habit = te_nb[ALT_SIGNALS].mean(axis=1).to_numpy()
        by_quintile = pd.Series(y).groupby(pd.qcut(habit, 5, labels=False)).mean()
        rows.append(
            {
                "scenario": name,
                "signal_auc": float(
                    np.mean([ev.roc_auc_score(y, -te_nb[signal].to_numpy()) for signal in ALT_SIGNALS])
                ),
                "signals_only_quintile_ratio": float(by_quintile.iloc[0] / by_quintile.iloc[-1]),
                "auc_no_bureau": float(ev.roc_auc_score(y, p)),
                **ev.policy_row(p <= cutoffs["approve"], y, te_nb["loan"].to_numpy()),
            }
        )
    return rows


def period(frame: pd.DataFrame) -> dict:
    return {
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "rows": int(len(frame)),
        "no_bureau_rows": int(frame["no_bureau"].sum()),
        "default_rate": float(frame["default"].mean()),
    }


def worst_region_ratio(booster, features, val, test) -> float:
    """Lowest regional approval as a share of the highest, using each variant's own cutoff fixed on validation."""
    val_nb, test_nb = val[val["no_bureau"]], test[test["no_bureau"]]
    target = val.loc[val["bureau_low_risk"], "default"].mean()
    cutoff = ev.pick_threshold(booster.predict(val_nb[features]), val_nb["default"].to_numpy(), target)
    audit = ev.group_audit(
        test_nb["region"].to_numpy(), booster.predict(test_nb[features]) <= cutoff, test_nb["default"].to_numpy()
    )
    return min(row["ratio"] for row in audit)


def logistic_baseline(train, test) -> dict:
    """The same inputs in a plain linear model, to show what the gradient boosting adds."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        train[FEATURES], train["default"]
    )
    p, y = model.predict_proba(test[FEATURES])[:, 1], test["default"].to_numpy()
    no_bureau, rated = test["no_bureau"].to_numpy(), test["bureau_rated"].to_numpy()
    return {
        "auc_no_bureau": float(ev.roc_auc_score(y[no_bureau], p[no_bureau])),
        "auc_rated": float(ev.roc_auc_score(y[rated], p[rated])),
    }


def variant_table(train, val, test) -> list[dict]:
    no_bureau, rated = test["no_bureau"], test["bureau_rated"]
    rows = []
    for name, features in VARIANTS.items():
        booster = fit(train, val, features)
        p = booster.predict(test[features])
        y = test["default"].to_numpy()
        rows.append(
            {
                "name": name,
                "auc_no_bureau": float(ev.roc_auc_score(y[no_bureau], p[no_bureau])),
                "auc_rated": float(ev.roc_auc_score(y[rated], p[rated])),
                "worst_region_ratio": worst_region_ratio(booster, features, val, test),
            }
        )
    return rows


def importance(booster: lgb.Booster, frame: pd.DataFrame) -> list[dict]:
    """Average size of each factor's effect on the score, in points."""
    sample = frame.sample(min(10_000, len(frame)), random_state=7)[FEATURES]
    contributions = booster.predict(sample, pred_contrib=True)[:, :-1]
    points = abs(points_from_margin(0) - points_from_margin(1))
    rows = []
    for factor, members in FACTOR_GROUPS.items():
        columns = [FEATURES.index(member) for member in members]
        rows.append({"factor": factor, "points": float(np.abs(contributions[:, columns].sum(axis=1)).mean() * points)})
    return sorted(rows, key=lambda row: -row["points"])


def main():
    started = time.time()
    frame = load()
    train, val, test = split(frame)
    train, val, test = encode_channels(train, val, test)

    booster = fit(train, val, FEATURES)
    platt = calibrate(booster, FEATURES, val)

    def predict(rows: pd.DataFrame) -> np.ndarray:
        return calibrated_risk(booster, FEATURES, platt, rows)

    cutoffs = fix_cutoffs(booster, FEATURES, platt, val)
    approve_at, high_risk_at = cutoffs["approve"], cutoffs["high_risk"]
    benchmark_val = float(val.loc[val["bureau_low_risk"], "default"].mean())

    ARTIFACTS.mkdir(exist_ok=True)
    booster.save_model(str(ARTIFACTS / "model.txt"))
    scenarios = alt_scenarios(train, val, test)
    tiers = {
        name: {
            "better": float(train[name].quantile(0.25)),
            "average": float(train["default"].mean()),
            "worse": float(train[name].quantile(0.75)),
        }
        for name in CHANNEL
    }
    meta = {
        "features": FEATURES,
        "platt": platt,
        "thresholds": cutoffs,
        "tiers": tiers,
        "price_range": [int(frame["vehicle_price"].min()), int(frame["vehicle_price"].max())],
        "ltv_range": [float(frame["ltv_pct"].min()), float(frame["ltv_pct"].max())],
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # October, the month the model never trained on.
    test_nb = test[test["no_bureau"]]
    y_all, p_all = test["default"].to_numpy(), predict(test)
    y_nb, p_nb, loan_nb = test_nb["default"].to_numpy(), predict(test_nb), test_nb["loan"].to_numpy()
    approved = p_nb <= approve_at
    benchmark = float(test.loc[test["bureau_low_risk"], "default"].mean())
    high = p_nb > high_risk_at
    review = ~approved & ~high

    rated = test[test["bureau_rated"]]
    y_rated, p_rated, bureau_score = (
        rated["default"].to_numpy(),
        predict(rated),
        -rated["PERFORM_CNS.SCORE"].to_numpy(dtype=float),
    )
    deciles = pd.Series(y_nb).groupby(pd.qcut(p_nb, 10, labels=False)).mean().tolist()
    ltv_nb = test_nb["ltv_pct"].to_numpy()
    grid = np.quantile(p_nb, np.linspace(0.02, 1.0, 50))
    curve = ev.policy_curve(p_nb, y_nb, loan_nb, grid)
    for row in curve:
        row["ltv_rule_default"] = ev.rule_default_at(ltv_nb, y_nb, row["share"])

    age_band = pd.cut(
        test_nb["age"], [0, 25, 35, 45, 120], labels=["Under 25", "25 to 34", "35 to 44", "45 and over"], right=False
    )
    monthly = (
        pd.DataFrame({"m": test["date"].dt.isocalendar().week.astype(int), "p": p_all, "y": y_all})
        .groupby("m")
        .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    )

    report = {
        "population": {
            "rows": int(len(frame)),
            "no_bureau": int(frame["no_bureau"].sum()),
            "no_bureau_share": float(frame["no_bureau"].mean()),
            "default_no_bureau": float(frame.loc[frame["no_bureau"], "default"].mean()),
            "default_with_history": float(frame.loc[~frame["no_bureau"], "default"].mean()),
        },
        "periods": {"train": period(train), "validation": period(val), "test": period(test)},
        "headline": {
            **ev.policy_row(approved, y_nb, loan_nb),
            "benchmark_default": benchmark,
            "benchmark_default_validation": benchmark_val,
            "no_bureau_default": float(y_nb.mean()),
            "not_approved_default": float(y_nb[~approved].mean()),
            "ltv_rule_default": ev.rule_default_at(ltv_nb, y_nb, float(approved.mean())),
            "applicants": int(len(y_nb)),
            "classification": ev.classification_at(approved, y_nb),
        },
        "bands": [
            {"band": "Approve", **ev.policy_row(approved, y_nb, loan_nb)},
            {"band": "Human review", **ev.policy_row(review, y_nb, loan_nb)},
            {"band": "High risk", **ev.policy_row(high, y_nb, loan_nb)},
        ],
        "auc": {
            "no_bureau": ev.bootstrap_auc(y_nb, p_nb),
            "all": ev.bootstrap_auc(y_all, p_all),
            "rated_zerofile": ev.bootstrap_auc(y_rated, p_rated),
            "rated_bureau_score": ev.bootstrap_auc(y_rated, bureau_score),
            "rated_gain": ev.bootstrap_auc_gain(y_rated, p_rated, bureau_score),
        },
        "ks_no_bureau": ev.ks_statistic(y_nb, p_nb),
        "brier_no_bureau": ev.brier(y_nb, p_nb),
        "brier_base_rate_no_bureau": ev.brier(y_nb, np.full(len(y_nb), y_nb.mean())),
        "ece_no_bureau": ev.expected_calibration_error(y_nb, p_nb),
        "calibration": ev.calibration_table(y_nb, p_nb),
        "weekly_drift": [
            {"week": int(r.Index), "predicted": float(r.predicted), "actual": float(r.actual), "n": int(r.n)}
            for r in monthly.itertuples()
        ],
        "risk_deciles": [float(value) for value in deciles],
        "policy_curve": curve,
        "variants": variant_table(train, val, test),
        "logistic_baseline": logistic_baseline(train, test),
        "fairness": {
            "Age band": ev.group_audit(age_band.astype(str).to_numpy(), approved, y_nb),
            "Employment": ev.group_audit(test_nb["employment"].astype(str).to_numpy(), approved, y_nb),
            "Region": ev.group_audit(test_nb["region"].to_numpy(), approved, y_nb),
        },
        "regional_floor": ev.regional_floor(p_nb, y_nb, loan_nb, test_nb["region"].to_numpy(), approved),
        "alt_scenarios": scenarios,
        "importance": importance(booster, test_nb),
        "model": {"trees": int(booster.best_iteration), "features": len(FEATURES)},
    }

    scorer = Scorer()
    sample = Applicant(vehicle_price=70_000, down_payment=20_000, employment="Self employed")
    timings = []
    for _ in range(200):
        tick = time.perf_counter()
        scorer.score(sample)
        timings.append((time.perf_counter() - tick) * 1000)
    report["latency_ms_p50"] = float(np.median(timings))

    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    headline = report["headline"]
    print(f"Finished in {time.time() - started:.0f}s. Trees: {booster.best_iteration}")
    print(f"AUC no-bureau {report['auc']['no_bureau']}")
    print(
        f"AUC rated: ZeroFile {report['auc']['rated_zerofile']['value']:.4f}, bureau score {report['auc']['rated_bureau_score']['value']:.4f}, gain {report['auc']['rated_gain']}"
    )
    print(
        f"Approved {headline['share']:.1%} of no-bureau ({headline['people']:,}, Rs {headline['crore']:.1f} crore), default {headline['default']:.2%}, benchmark {benchmark:.2%}, LTV rule {headline['ltv_rule_default']:.2%}"
    )
    print(pd.DataFrame(report["variants"]).round(4).to_string())
    print(pd.DataFrame(report["bands"]).round(4).to_string())
    print(pd.DataFrame(report["fairness"]["Age band"]).round(3).to_string())
    print(pd.DataFrame(report["fairness"]["Employment"]).round(3).to_string())
    print(pd.DataFrame(report["fairness"]["Region"]).round(3).to_string())
    print("Regional floor", report["regional_floor"])
    print(pd.DataFrame(scenarios).round(4).to_string())
    print("Deciles", [round(v, 3) for v in deciles])
    print(
        "Calibration",
        {
            k: round(report[k], 4)
            for k in ("ks_no_bureau", "brier_no_bureau", "brier_base_rate_no_bureau", "ece_no_bureau")
        },
    )
    print("Weekly", report["weekly_drift"])
    print(
        "Importance",
        report["importance"],
        "Classification",
        headline["classification"],
        "Latency ms",
        report["latency_ms_p50"],
    )


if __name__ == "__main__":
    main()
