"""Download, validate and split the L&T vehicle-loan dataset. Run `python -m zerofile.data` to fetch it."""
import base64
import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PAGE = "https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/mamtadhaker/lt-vehicle-loan-default-prediction"

NO_BUREAU = "No Bureau History Available"
VAL_START = pd.Timestamp("2018-09-24")
TEST_START = pd.Timestamp("2018-10-01")

BUREAU_NUMERIC = [
    "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE",
    "PRI.SANCTIONED.AMOUNT", "PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS",
    "SEC.CURRENT.BALANCE", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS",
    "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "NO.OF_INQUIRIES",
]
BUREAU_FEATURES = BUREAU_NUMERIC + ["avg_account_months", "history_months"]
REQUIRED_COLUMNS = set(BUREAU_NUMERIC) | {
    "disbursed_amount", "asset_cost", "branch_id", "supplier_id", "manufacturer_id", "Date.of.Birth", "Employment.Type",
    "DisbursalDate", "State_ID", "Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "Passport_flag",
    "PERFORM_CNS.SCORE.DESCRIPTION", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH", "loan_default",
}


def _load_env():
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def data_dir() -> Path:
    _load_env()
    return Path(os.environ.get("ZEROFILE_DATA_DIR") or ROOT / "data")


def download() -> Path:
    """Fetch train.csv once. Later calls return the existing file."""
    target = data_dir() / "train.csv"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "zerofile"})
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        token = base64.b64encode(f"{user}:{key}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not download the dataset ({exc}).\n"
            f"Download it from {DATASET_PAGE} and put train.csv in {target.parent}, "
            "or add Kaggle credentials to .env."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive, archive.open("train.csv") as source:
        target.write_bytes(source.read())
    return target


def _months(text: pd.Series) -> pd.Series:
    """Turn '1yrs 11mon' into 23."""
    parts = text.str.extract(r"(\d+)yrs (\d+)mon").astype(int)
    return parts[0] * 12 + parts[1]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep what the project needs. Date of birth, pincode, customer ID and employee code stop here."""
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    date = pd.to_datetime(raw["DisbursalDate"], format="%d-%m-%y")
    born = pd.to_datetime(raw["Date.of.Birth"], format="%d-%m-%y")
    # Two-digit years: pandas reads 68 as 2068, which for a borrower means 1968.
    born = born.where(born < date, born - pd.DateOffset(years=100))
    description = raw["PERFORM_CNS.SCORE.DESCRIPTION"]
    frame = pd.DataFrame({
        "date": date,
        "default": raw["loan_default"],
        "loan": raw["disbursed_amount"],
        "vehicle_price": raw["asset_cost"],
        "ltv_pct": 100 * raw["disbursed_amount"] / raw["asset_cost"],
        "log_vehicle_price": np.log(raw["asset_cost"]),
        "self_employed": raw["Employment.Type"].map({"Self employed": 1.0, "Salaried": 0.0}),
        "employment": raw["Employment.Type"].fillna("Not stated"),
        "aadhaar": raw["Aadhar_flag"],
        "pan": raw["PAN_flag"],
        "voter_id": raw["VoterID_flag"],
        "driving_licence": raw["Driving_flag"],
        "passport": raw["Passport_flag"],
        "brand": raw["manufacturer_id"],
        "dealer": raw["supplier_id"],
        "branch": raw["branch_id"],
        "region": raw["State_ID"],
        "age": (date - born).dt.days / 365.25,
        "no_bureau": description == NO_BUREAU,
        "bureau_rated": description.str.match(r"^[A-M]-"),
        "bureau_low_risk": description.str.match(r"^[A-G]-"),
        "avg_account_months": _months(raw["AVERAGE.ACCT.AGE"]),
        "history_months": _months(raw["CREDIT.HISTORY.LENGTH"]),
    })
    return pd.concat([frame, raw[BUREAU_NUMERIC]], axis=1)


def load() -> pd.DataFrame:
    path = data_dir() / "train.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m zerofile.data")
    return prepare(pd.read_csv(path))


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on the past, tune on late September, test on October. Default rates drift by month, so never split randomly."""
    return (
        frame[frame["date"] < VAL_START],
        frame[(frame["date"] >= VAL_START) & (frame["date"] < TEST_START)],
        frame[frame["date"] >= TEST_START],
    )


if __name__ == "__main__":
    print(f"Dataset ready at {download()}")
