"""Model features. Bureau data, age and region are deliberately left out of FEATURES."""
import numpy as np
import pandas as pd

LOAN = ["ltv_pct", "log_vehicle_price"]
PROFILE = ["self_employed", "aadhaar", "pan", "voter_id", "driving_licence", "passport"]
CHANNEL = ["brand_risk", "dealer_risk", "branch_risk"]
FEATURES = LOAN + PROFILE + CHANNEL
ENCODED = {"brand": "brand_risk", "dealer": "dealer_risk", "branch": "branch_risk", "region": "region_risk"}


def monotone_for(features: list[str]) -> list[int]:
    """A bigger loan against the same vehicle can never lower risk."""
    return [1 if name == "ltv_pct" else 0 for name in features]


def _shrunk_rates(frame: pd.DataFrame, column: str, prior: float, smoothing: int) -> pd.Series:
    stats = frame.groupby(column)["default"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


def encode_channels(train: pd.DataFrame, *later: pd.DataFrame, smoothing: int = 200, folds: int = 5, seed: int = 7):
    """Replace IDs with their historic default rate, shrunk towards the overall rate.

    Training rows get out-of-fold values so no row sees its own label. Later periods only see
    what training saw, and an unseen dealer gets the overall rate.
    """
    prior = train["default"].mean()
    fold = np.random.default_rng(seed).integers(0, folds, len(train))
    train = train.copy()
    later = [frame.copy() for frame in later]
    for column, name in ENCODED.items():
        values = np.full(len(train), prior)
        for k in range(folds):
            rates = _shrunk_rates(train[fold != k], column, prior, smoothing)
            values[fold == k] = train.loc[fold == k, column].map(rates).fillna(prior)
        train[name] = values
        rates = _shrunk_rates(train, column, prior, smoothing)
        for frame in later:
            frame[name] = frame[column].map(rates).fillna(prior)
    return (train, *later)
