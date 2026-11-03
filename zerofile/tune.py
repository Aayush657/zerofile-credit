"""Does tuning help the real-data model? `python -m zerofile.tune` scans settings and writes artifacts/tuning.json.

The shipped settings are not changed by this scan. It exists to show that AUC is limited by the data, not the settings.
"""
import itertools
import json

import lightgbm as lgb

from zerofile import evaluate as ev
from zerofile.data import load, split
from zerofile.features import FEATURES, encode_channels, monotone_for
from zerofile.scoring import ARTIFACTS
from zerofile.train import PARAMS

GRID = {"num_leaves": (7, 15, 31, 63), "min_child_samples": (100, 200, 500, 1000), "colsample_bytree": (0.6, 0.8)}


def main():
    train, val, test = split(load())
    train, val, test = encode_channels(train, val, test)
    no_bureau = test["no_bureau"].to_numpy()
    y = test["default"].to_numpy()
    rows = []
    for values in itertools.product(*GRID.values()):
        settings = dict(zip(GRID, values))
        booster = lgb.train(
            {**PARAMS, **settings, "monotone_constraints": monotone_for(FEATURES)},
            lgb.Dataset(train[FEATURES], train["default"]),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(val[FEATURES], val["default"])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        rows.append({
            **settings,
            "auc_validation": float(ev.roc_auc_score(val["default"], booster.predict(val[FEATURES]))),
            "auc_test_no_bureau": float(ev.roc_auc_score(y[no_bureau], booster.predict(test[FEATURES])[no_bureau])),
        })
    tests = [row["auc_test_no_bureau"] for row in rows]
    summary = {"configurations": len(rows), "auc_test_low": min(tests), "auc_test_high": max(tests), "best_on_validation": max(rows, key=lambda row: row["auc_validation"]), "rows": rows}
    (ARTIFACTS / "tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({key: summary[key] for key in ("configurations", "auc_test_low", "auc_test_high", "best_on_validation")})


if __name__ == "__main__":
    main()
