"""Runtime scoring: risk, points breakdown, plain-language reasons and recourse. Needs only NumPy and LightGBM."""
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# 100 points halve the odds of default, and 600 points means a 20% risk before calibration.
POINTS_PER_LOG_ODDS = 100 / math.log(2)
ANCHOR_POINTS = 600
ANCHOR_LOG_ODDS = math.log(0.2 / 0.8)

FACTOR_GROUPS = {
    "Down payment": ["ltv_pct"],
    "Vehicle price": ["log_vehicle_price"],
    "Employment": ["self_employed"],
    "ID documents": ["aadhaar", "pan", "voter_id", "driving_licence", "passport"],
    "Vehicle brand": ["brand_risk"],
    "Dealer record": ["dealer_risk"],
    "Branch record": ["branch_risk"],
}
EMPLOYMENT_CODES = {"Salaried": 0.0, "Self employed": 1.0, "Not stated": float("nan")}
TIER_LABELS = {"better": "better than typical", "typical": "typical", "worse": "worse than typical"}
RECOURSE_STEP = 500  # Rupees between the options tried, rounded the way a person would pay.


def points_from_margin(margin):
    return ANCHOR_POINTS - POINTS_PER_LOG_ODDS * (margin - ANCHOR_LOG_ODDS)


@dataclass(frozen=True)
class Applicant:
    vehicle_price: int
    down_payment: int
    employment: str = "Not stated"
    aadhaar: bool = True
    pan: bool = False
    voter_id: bool = False
    driving_licence: bool = False
    passport: bool = False
    brand: str = "typical"
    dealer: str = "typical"
    branch: str = "typical"

    @property
    def loan(self) -> int:
        return self.vehicle_price - self.down_payment

    @property
    def ltv_pct(self) -> float:
        return 100 * self.loan / self.vehicle_price


@dataclass(frozen=True)
class Result:
    risk: float
    points: float
    band: str
    base_points: float
    factors: dict
    reasons: list
    warnings: list


class Scorer:
    def __init__(self, directory: Path = ARTIFACTS):
        self.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.features = self.meta["features"]

    def _row(self, a: Applicant) -> list:
        values = {
            "ltv_pct": a.ltv_pct,
            "log_vehicle_price": math.log(a.vehicle_price),
            "self_employed": EMPLOYMENT_CODES[a.employment],
            "aadhaar": float(a.aadhaar),
            "pan": float(a.pan),
            "voter_id": float(a.voter_id),
            "driving_licence": float(a.driving_licence),
            "passport": float(a.passport),
            "brand_risk": self.meta["tiers"]["brand_risk"][a.brand],
            "dealer_risk": self.meta["tiers"]["dealer_risk"][a.dealer],
            "branch_risk": self.meta["tiers"]["branch_risk"][a.branch],
        }
        return [values[name] for name in self.features]

    def _risk(self, margin):
        platt = self.meta["platt"]
        return 1 / (1 + np.exp(-(platt["slope"] * margin + platt["intercept"])))

    def _band(self, risk: float) -> str:
        thresholds = self.meta["thresholds"]
        if risk <= thresholds["approve"]:
            return "Approve"
        return "High risk" if risk > thresholds["high_risk"] else "Human review"

    def validate(self, a: Applicant) -> None:
        if a.vehicle_price <= 0:
            raise ValueError("Vehicle price must be more than zero.")
        if not 0 <= a.down_payment < a.vehicle_price:
            raise ValueError("Down payment must be at least zero and less than the vehicle price.")
        if a.employment not in EMPLOYMENT_CODES:
            raise ValueError(f"Unknown employment type: {a.employment}")
        for tier in (a.brand, a.dealer, a.branch):
            if tier not in TIER_LABELS:
                raise ValueError(f"Unknown track-record level: {tier}")

    def score(self, a: Applicant) -> Result:
        self.validate(a)
        contributions = self.booster.predict(np.array([self._row(a)], dtype=float), pred_contrib=True)[0]
        margin = float(contributions.sum())
        risk = float(self._risk(margin))
        factors = {}
        for factor, members in FACTOR_GROUPS.items():
            # Lower log-odds of default means more points, so the sign flips.
            factors[factor] = -POINTS_PER_LOG_ODDS * float(sum(contributions[self.features.index(m)] for m in members))
        base = float(points_from_margin(contributions[-1]))
        return Result(risk, float(points_from_margin(margin)), self._band(risk), base, factors, self._reasons(a, factors), self._warnings(a))

    def _warnings(self, a: Applicant) -> list:
        low_price, high_price = self.meta["price_range"]
        low_ltv, high_ltv = self.meta["ltv_range"]
        notes = []
        if not low_price <= a.vehicle_price <= high_price:
            notes.append(f"The vehicle price is outside the ₹{low_price:,} to ₹{high_price:,} range the model learned from, so treat this result with caution.")
        if not low_ltv <= a.ltv_pct <= high_ltv:
            notes.append(f"The loan is {a.ltv_pct:.0f}% of the vehicle price, outside the {low_ltv:.0f}% to {high_ltv:.0f}% the model learned from, so treat this result with caution.")
        return notes

    def _reasons(self, a: Applicant, factors: dict) -> list:
        documents = [name for name, given in (("Aadhaar", a.aadhaar), ("PAN", a.pan), ("Voter ID", a.voter_id), ("Driving licence", a.driving_licence), ("Passport", a.passport)) if given]
        details = {
            "Down payment": f"Paying ₹{a.down_payment:,} down, {100 - a.ltv_pct:.0f}% of the vehicle price.",
            "Vehicle price": f"The vehicle costs ₹{a.vehicle_price:,}.",
            "Employment": f"Employment type: {a.employment.lower()}.",
            "ID documents": "Documents given: " + (", ".join(documents) if documents else "none") + ".",
            "Vehicle brand": f"This vehicle brand's repayment record is {TIER_LABELS[a.brand]}.",
            "Dealer record": f"This dealer's repayment record is {TIER_LABELS[a.dealer]}.",
            "Branch record": f"This branch's repayment record is {TIER_LABELS[a.branch]}.",
        }
        ordered = sorted(factors.items(), key=lambda item: -abs(item[1]))
        return [{"factor": factor, "points": points, "detail": details[factor]} for factor, points in ordered]

    def _bands_for(self, applicants: list) -> np.ndarray:
        rows = np.array([self._row(a) for a in applicants], dtype=float)
        return self._risk(self.booster.predict(rows, raw_score=True))

    def recourse(self, a: Applicant) -> list:
        """The smallest change to the loan itself that reaches the approve band. Identity, dealer and brand are never suggested."""
        self.validate(a)
        cutoff = self.meta["thresholds"]["approve"]
        if self._bands_for([a])[0] <= cutoff:
            return []
        options = []
        # The monotonic LTV constraint means a bigger down payment can only help, so the first pass is the smallest.
        most_down = int(a.vehicle_price * (1 - self.meta["ltv_range"][0] / 100))
        downs = list(range(a.down_payment + RECOURSE_STEP, most_down + 1, RECOURSE_STEP))
        if downs:
            risks = self._bands_for([replace(a, down_payment=down) for down in downs])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                down = downs[hits[0]]
                options.append({"kind": "down_payment", "down_payment": down, "extra": down - a.down_payment, "risk": float(risks[hits[0]])})
        low_price = self.meta["price_range"][0]
        prices = list(range(a.vehicle_price - RECOURSE_STEP, max(low_price, a.down_payment + 1) - 1, -RECOURSE_STEP))
        if prices:
            risks = self._bands_for([replace(a, vehicle_price=price) for price in prices])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                price = prices[hits[0]]
                options.append({"kind": "cheaper_vehicle", "vehicle_price": price, "saving": a.vehicle_price - price, "risk": float(risks[hits[0]])})
        return options
