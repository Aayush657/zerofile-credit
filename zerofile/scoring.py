"""Runtime scoring: risk, points breakdown, reasons, loan range and recourse. Needs only NumPy and LightGBM."""
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
    "Vehicle brand": ["brand_risk"],
    "Dealer record": ["dealer_risk"],
    "Branch record": ["branch_risk"],
}
ALT_FACTOR_GROUPS = {"Utility bills": ["bills_on_time"], "Mobile top-ups": ["topup_regularity"], "Income regularity": ["income_regularity"]}
EMPLOYMENT_CODES = {"Salaried": 0.0, "Self employed": 1.0, "Not stated": float("nan")}
TIER_LABELS = {"better": "better than average", "average": "average", "worse": "worse than average"}
RECOURSE_STEP = 500  # Rupees between the options tried, rounded the way a person would pay.


def points_from_margin(margin):
    return ANCHOR_POINTS - POINTS_PER_LOG_ODDS * (margin - ANCHOR_LOG_ODDS)


@dataclass(frozen=True)
class Applicant:
    vehicle_price: int
    down_payment: int
    employment: str = "Not stated"
    brand: str = "average"
    dealer: str = "average"
    branch: str = "average"
    # Simulated signals as shares from 0 to 1: all three or none. See altdata.py.
    bills_on_time: float | None = None
    topup_regularity: float | None = None
    income_regularity: float | None = None

    @property
    def loan(self) -> int:
        return self.vehicle_price - self.down_payment

    @property
    def ltv_pct(self) -> float:
        return 100 * self.loan / self.vehicle_price

    @property
    def simulated(self) -> bool:
        return self.bills_on_time is not None


@dataclass(frozen=True)
class Result:
    risk: float
    points: float
    band: str
    base_points: float
    factors: dict
    reasons: list
    warnings: list
    simulated: bool = False


class Scorer:
    def __init__(self, directory: Path = ARTIFACTS):
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.base = self._load(directory / "model.txt", self.meta)
        self.alt = self._load(directory / "alt_model.txt", self.meta["alt"])

    @staticmethod
    def _load(path: Path, meta: dict) -> dict:
        return {"booster": lgb.Booster(model_file=str(path)), "features": meta["features"], "platt": meta["platt"], "thresholds": meta["thresholds"]}

    def spec_for(self, a: Applicant) -> dict:
        return self.alt if a.simulated else self.base

    def _row(self, a: Applicant) -> list:
        values = {
            "ltv_pct": a.ltv_pct,
            "log_vehicle_price": math.log(a.vehicle_price),
            "self_employed": EMPLOYMENT_CODES[a.employment],
            "brand_risk": self.meta["tiers"]["brand_risk"][a.brand],
            "dealer_risk": self.meta["tiers"]["dealer_risk"][a.dealer],
            "branch_risk": self.meta["tiers"]["branch_risk"][a.branch],
            "bills_on_time": a.bills_on_time,
            "topup_regularity": a.topup_regularity,
            "income_regularity": a.income_regularity,
        }
        return [values[name] for name in self.spec_for(a)["features"]]

    @staticmethod
    def _risk(margin, spec: dict):
        return 1 / (1 + np.exp(-(spec["platt"]["slope"] * margin + spec["platt"]["intercept"])))

    def points_for_risk(self, risk: float, spec: dict | None = None) -> float:
        platt = (spec or self.base)["platt"]
        return float(points_from_margin((math.log(risk / (1 - risk)) - platt["intercept"]) / platt["slope"]))

    def _band(self, risk: float, spec: dict | None = None) -> str:
        thresholds = (spec or self.base)["thresholds"]
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
        signals = (a.bills_on_time, a.topup_regularity, a.income_regularity)
        if any(signal is not None for signal in signals) and not all(signal is not None and 0 <= signal <= 1 for signal in signals):
            raise ValueError("Give all three simulated signals as shares from 0 to 1, or none of them.")

    def score(self, a: Applicant) -> Result:
        self.validate(a)
        spec = self.spec_for(a)
        contributions = spec["booster"].predict(np.array([self._row(a)], dtype=float), pred_contrib=True)[0]
        margin = float(contributions.sum())
        risk = float(self._risk(margin, spec))
        groups = FACTOR_GROUPS | ALT_FACTOR_GROUPS if a.simulated else FACTOR_GROUPS
        factors = {}
        for factor, members in groups.items():
            # Lower log-odds of default means more points, so the sign flips.
            factors[factor] = -POINTS_PER_LOG_ODDS * float(sum(contributions[spec["features"].index(m)] for m in members))
        base = float(points_from_margin(contributions[-1]))
        return Result(risk, float(points_from_margin(margin)), self._band(risk, spec), base, factors, self._reasons(a, factors), self._warnings(a), a.simulated)

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
        details = {
            "Down payment": f"Paying ₹{a.down_payment:,} down, {100 - a.ltv_pct:.0f}% of the vehicle price.",
            "Vehicle price": f"The vehicle costs ₹{a.vehicle_price:,}.",
            "Employment": f"Employment type: {a.employment.lower()}.",
            "Vehicle brand": f"This vehicle brand's repayment record is {TIER_LABELS[a.brand]}.",
            "Dealer record": f"This dealer's repayment record is {TIER_LABELS[a.dealer]}.",
            "Branch record": f"This branch's repayment record is {TIER_LABELS[a.branch]}.",
        }
        if a.simulated:
            details |= {
                "Utility bills": f"Simulated signal: {a.bills_on_time:.0%} of bills paid on time.",
                "Mobile top-ups": f"Simulated signal: {a.topup_regularity:.0%} of mobile top-ups on a regular schedule.",
                "Income regularity": f"Simulated signal: steady income in {a.income_regularity:.0%} of months.",
            }
        ordered = sorted(factors.items(), key=lambda item: -abs(item[1]))
        return [{"factor": factor, "points": points, "detail": details[factor]} for factor, points in ordered]

    def _risks_for(self, applicants: list) -> np.ndarray:
        spec = self.spec_for(applicants[0])
        rows = np.array([self._row(a) for a in applicants], dtype=float)
        return self._risk(spec["booster"].predict(rows, raw_score=True), spec)

    def loan_range(self, a: Applicant) -> dict:
        """The largest loan on this vehicle that stays in each band. Risk only rises with the loan, so the last pass is the limit."""
        self.validate(a)
        thresholds = self.spec_for(a)["thresholds"]
        low_ltv, high_ltv = self.meta["ltv_range"]
        lowest = math.ceil(a.vehicle_price * low_ltv / 100 / RECOURSE_STEP) * RECOURSE_STEP
        loans = list(range(lowest, int(a.vehicle_price * high_ltv / 100) + 1, RECOURSE_STEP))
        if not loans:
            return {"approve_up_to": None, "review_up_to": None}
        risks = self._risks_for([replace(a, down_payment=a.vehicle_price - loan) for loan in loans])
        limit = lambda cutoff: next((loans[i] for i in range(len(loans) - 1, -1, -1) if risks[i] <= cutoff), None)
        return {"approve_up_to": limit(thresholds["approve"]), "review_up_to": limit(thresholds["high_risk"])}

    def recourse(self, a: Applicant) -> list:
        """The smallest change to the loan itself that reaches the approve band. Identity, dealer and brand are never suggested."""
        self.validate(a)
        cutoff = self.spec_for(a)["thresholds"]["approve"]
        if self._risks_for([a])[0] <= cutoff:
            return []
        options = []
        # The monotonic LTV constraint means a bigger down payment can only help, so the first pass is the smallest.
        most_down = int(a.vehicle_price * (1 - self.meta["ltv_range"][0] / 100))
        downs = list(range(a.down_payment + RECOURSE_STEP, most_down + 1, RECOURSE_STEP))
        if downs:
            risks = self._risks_for([replace(a, down_payment=down) for down in downs])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                down = downs[hits[0]]
                options.append({"kind": "down_payment", "down_payment": down, "extra": down - a.down_payment, "risk": float(risks[hits[0]])})
        low_price = self.meta["price_range"][0]
        prices = list(range(a.vehicle_price - RECOURSE_STEP, max(low_price, a.down_payment + 1) - 1, -RECOURSE_STEP))
        if prices:
            risks = self._risks_for([replace(a, vehicle_price=price) for price in prices])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                price = prices[hits[0]]
                options.append({"kind": "cheaper_vehicle", "vehicle_price": price, "saving": a.vehicle_price - price, "risk": float(risks[hits[0]])})
        return options
