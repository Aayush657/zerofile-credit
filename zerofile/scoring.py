"""Runtime scoring for both models: risk, points breakdown, reasons, loan range and recourse. Needs only NumPy and LightGBM."""

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
WORKER_FACTOR_GROUPS = {
    "Utility bills": ["bills_on_time"],
    "Rent": ["rent_on_time"],
    "Mobile recharges": ["recharges_per_month"],
    "UPI use": ["upi_transactions", "upi_months"],
    "Income steadiness": ["income_swing"],
    "Loan burden": ["emi_to_income"],
}
# The worker model is far sharper, so its log-odds span more. This keeps its score inside 300 to 900 too.
WORKER_SCALE = {"per_log_odds": 48.0, "anchor": -1.2}
EMPLOYMENT_CODES = {"Salaried": 0.0, "Self employed": 1.0, "Not stated": float("nan")}
TIER_LABELS = {"better": "better than average", "average": "average", "worse": "worse than average"}
RECOURSE_STEP = 500  # Rupees between the options tried, rounded the way a person would pay.
WORKER_LOAN_STEP = 1_000
WORKER_LOAN_LIMITS = (5_000, 500_000)
# A policy guardrail, not something the model learned: the synthetic labels barely punish oversized loans.
AFFORDABILITY_CAP = 0.4


def points_from_margin(margin):
    return ANCHOR_POINTS - POINTS_PER_LOG_ODDS * (margin - ANCHOR_LOG_ODDS)


def net_value(crore, default, margin: float, loss: float):
    """Lender's value in crore: interest on loans that repay, minus losses on loans that default. Margin and loss are assumptions."""
    return crore * ((1 - default) * margin - default * loss)


@dataclass(frozen=True)
class Applicant:
    vehicle_price: int
    down_payment: int
    employment: str = "Not stated"
    brand: str = "average"
    dealer: str = "average"
    branch: str = "average"

    @property
    def loan(self) -> int:
        return self.vehicle_price - self.down_payment

    @property
    def ltv_pct(self) -> float:
        return 100 * self.loan / self.vehicle_price


@dataclass(frozen=True)
class Worker:
    """Shares run from 0 to 1. None means no record, such as someone who pays no rent or has never used UPI."""

    bills_on_time: float
    recharges_per_month: float
    monthly_income: int
    income_swing: float
    loan_amount: int
    tenure_months: int
    rent_on_time: float | None = None
    upi_transactions: float | None = None
    upi_months: float | None = None

    @property
    def emi(self) -> float:
        """Monthly instalment before interest."""
        return self.loan_amount / self.tenure_months

    @property
    def emi_to_income(self) -> float:
        return self.emi / self.monthly_income


@dataclass(frozen=True)
class Result:
    risk: float
    points: float
    band: str
    base_points: float
    factors: dict
    reasons: list
    warnings: list


class _Model:
    per_log_odds = POINTS_PER_LOG_ODDS
    anchor = ANCHOR_LOG_ODDS

    def __init__(self, model_path: Path, meta_path: Path, groups: dict):
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=str(model_path))
        self.features = self.meta["features"]
        self.groups = groups

    def _points(self, margin):
        return ANCHOR_POINTS - self.per_log_odds * (margin - self.anchor)

    def _risk(self, margin):
        platt = self.meta["platt"]
        return 1 / (1 + np.exp(-(platt["slope"] * margin + platt["intercept"])))

    def points_for_risk(self, risk: float) -> float:
        platt = self.meta["platt"]
        return float(self._points((math.log(risk / (1 - risk)) - platt["intercept"]) / platt["slope"]))

    def _band(self, risk: float) -> str:
        thresholds = self.meta["thresholds"]
        if risk <= thresholds["approve"]:
            return "Approve"
        return "High risk" if risk > thresholds["high_risk"] else "Human review"

    def _explain(self, row: list) -> tuple:
        contributions = self.booster.predict(np.array([row], dtype=float), pred_contrib=True)[0]
        margin = float(contributions.sum())
        # Lower log-odds of default means more points, so the sign flips.
        factors = {
            factor: -self.per_log_odds * float(sum(contributions[self.features.index(member)] for member in members))
            for factor, members in self.groups.items()
        }
        risk = float(self._risk(margin))
        return risk, float(self._points(margin)), self._band(risk), float(self._points(contributions[-1])), factors

    def _risks(self, rows: list) -> np.ndarray:
        return self._risk(self.booster.predict(np.array(rows, dtype=float), raw_score=True))

    @staticmethod
    def _ordered(factors: dict, details: dict) -> list:
        ordered = sorted(factors.items(), key=lambda item: -abs(item[1]))
        return [{"factor": factor, "points": points, "detail": details[factor]} for factor, points in ordered]


class Scorer(_Model):
    """The vehicle-loan model, trained on real repayment outcomes."""

    def __init__(self, directory: Path = ARTIFACTS):
        super().__init__(directory / "model.txt", directory / "meta.json", FACTOR_GROUPS)

    def _row(self, a: Applicant) -> list:
        values = {
            "ltv_pct": a.ltv_pct,
            "log_vehicle_price": math.log(a.vehicle_price),
            "self_employed": EMPLOYMENT_CODES[a.employment],
            "brand_risk": self.meta["tiers"]["brand_risk"][a.brand],
            "dealer_risk": self.meta["tiers"]["dealer_risk"][a.dealer],
            "branch_risk": self.meta["tiers"]["branch_risk"][a.branch],
        }
        return [values[name] for name in self.features]

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
        risk, points, band, base, factors = self._explain(self._row(a))
        return Result(risk, points, band, base, factors, self._reasons(a, factors), self._warnings(a))

    def _warnings(self, a: Applicant) -> list:
        low_price, high_price = self.meta["price_range"]
        low_ltv, high_ltv = self.meta["ltv_range"]
        notes = []
        if not low_price <= a.vehicle_price <= high_price:
            notes.append(
                f"The vehicle price is outside the ₹{low_price:,} to ₹{high_price:,} range the model learned from, so treat this result with caution."
            )
        if not low_ltv <= a.ltv_pct <= high_ltv:
            notes.append(
                f"The loan is {a.ltv_pct:.0f}% of the vehicle price, outside the {low_ltv:.0f}% to {high_ltv:.0f}% the model learned from, so treat this result with caution."
            )
        return notes

    def _reasons(self, a: Applicant, factors: dict) -> list:
        return self._ordered(
            factors,
            {
                "Down payment": f"Paying ₹{a.down_payment:,} down, {100 - a.ltv_pct:.0f}% of the vehicle price.",
                "Vehicle price": f"The vehicle costs ₹{a.vehicle_price:,}.",
                "Employment": f"Employment type: {a.employment.lower()}.",
                "Vehicle brand": f"This vehicle brand's repayment record is {TIER_LABELS[a.brand]}.",
                "Dealer record": f"This dealer's repayment record is {TIER_LABELS[a.dealer]}.",
                "Branch record": f"This branch's repayment record is {TIER_LABELS[a.branch]}.",
            },
        )

    def _risks_for(self, applicants: list) -> np.ndarray:
        return self._risks([self._row(a) for a in applicants])

    def loan_range(self, a: Applicant) -> dict:
        """The largest loan on this vehicle that stays in each band. Risk only rises with the loan, so the last pass is the limit."""
        self.validate(a)
        thresholds = self.meta["thresholds"]
        low_ltv, high_ltv = self.meta["ltv_range"]
        lowest = math.ceil(a.vehicle_price * low_ltv / 100 / RECOURSE_STEP) * RECOURSE_STEP
        loans = list(range(lowest, int(a.vehicle_price * high_ltv / 100) + 1, RECOURSE_STEP))
        if not loans:
            return {"approve_up_to": None, "review_up_to": None}
        risks = self._risks_for([replace(a, down_payment=a.vehicle_price - loan) for loan in loans])

        def limit(cutoff):
            return next((loans[i] for i in range(len(loans) - 1, -1, -1) if risks[i] <= cutoff), None)

        return {"approve_up_to": limit(thresholds["approve"]), "review_up_to": limit(thresholds["high_risk"])}

    def recourse(self, a: Applicant) -> list:
        """The smallest change to the loan itself that reaches the approve band. Identity, dealer and brand are never suggested."""
        self.validate(a)
        cutoff = self.meta["thresholds"]["approve"]
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
                options.append(
                    {
                        "kind": "down_payment",
                        "down_payment": down,
                        "extra": down - a.down_payment,
                        "risk": float(risks[hits[0]]),
                    }
                )
        low_price = self.meta["price_range"][0]
        prices = list(range(a.vehicle_price - RECOURSE_STEP, max(low_price, a.down_payment + 1) - 1, -RECOURSE_STEP))
        if prices:
            risks = self._risks_for([replace(a, vehicle_price=price) for price in prices])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                price = prices[hits[0]]
                options.append(
                    {
                        "kind": "cheaper_vehicle",
                        "vehicle_price": price,
                        "saving": a.vehicle_price - price,
                        "risk": float(risks[hits[0]]),
                    }
                )
        return options


class WorkerScorer(_Model):
    """The alternative-data model. Its training data is synthetic, so every result is a demonstration."""

    per_log_odds = WORKER_SCALE["per_log_odds"]
    anchor = WORKER_SCALE["anchor"]

    def __init__(self, directory: Path = ARTIFACTS):
        super().__init__(directory / "worker_model.txt", directory / "worker_meta.json", WORKER_FACTOR_GROUPS)

    def _row(self, w: Worker) -> list:
        values = {
            "bills_on_time": w.bills_on_time,
            "rent_on_time": w.rent_on_time,
            "recharges_per_month": w.recharges_per_month,
            "upi_transactions": w.upi_transactions,
            "upi_months": w.upi_months,
            "income_swing": w.income_swing,
            "emi_to_income": w.emi_to_income,
        }
        return [values[name] for name in self.features]

    def validate(self, w: Worker) -> None:
        if w.monthly_income <= 0:
            raise ValueError("Monthly income must be more than zero.")
        if w.loan_amount <= 0 or w.tenure_months <= 0:
            raise ValueError("Loan amount and repayment period must be more than zero.")
        for share in (w.bills_on_time, w.rent_on_time, w.income_swing):
            if share is not None and not 0 <= share <= 1:
                raise ValueError("Shares must be between 0 and 1.")
        if (
            min(value for value in (w.recharges_per_month, w.upi_transactions, w.upi_months, 0) if value is not None)
            < 0
        ):
            raise ValueError("Counts cannot be negative.")
        if (w.upi_transactions is None) != (w.upi_months is None):
            raise ValueError("Give both UPI values, or neither for someone who does not use UPI.")

    def score(self, w: Worker) -> Result:
        self.validate(w)
        risk, points, band, base, factors = self._explain(self._row(w))
        return Result(risk, points, band, base, factors, self._reasons(w, factors), self._warnings(w))

    def _warnings(self, w: Worker) -> list:
        notes = []
        if w.upi_transactions is None:
            notes.append(
                "This worker has no UPI history. The training data treats that as a strong risk signal, which may reflect the dataset's assumptions more than this person, so a person should review this result."
            )
        if w.emi_to_income > AFFORDABILITY_CAP:
            notes.append(
                f"The instalment is {w.emi_to_income:.0%} of income, above the {AFFORDABILITY_CAP:.0%} affordability guardrail, so a person should check this loan whatever the score says."
            )
        return notes

    def _reasons(self, w: Worker, factors: dict) -> list:
        return self._ordered(
            factors,
            {
                "Utility bills": f"{w.bills_on_time:.0%} of utility bills paid on time.",
                "Rent": f"{w.rent_on_time:.0%} of rent paid on time."
                if w.rent_on_time is not None
                else "No rent record.",
                "Mobile recharges": f"{w.recharges_per_month:g} mobile recharges a month.",
                "UPI use": f"{w.upi_transactions:g} UPI payments a month, active for {w.upi_months:g} months."
                if w.upi_transactions is not None
                else "No UPI history.",
                "Income steadiness": f"Income swings about {w.income_swing:.0%} from month to month.",
                "Loan burden": f"Instalment of about ₹{w.emi:,.0f} a month, {w.emi_to_income:.0%} of income, before interest.",
            },
        )

    def loan_range(self, w: Worker) -> dict:
        """The largest affordable loan over this repayment period that stays in each band. Risk only rises with the instalment."""
        self.validate(w)
        thresholds = self.meta["thresholds"]
        low, high = WORKER_LOAN_LIMITS
        most = min(high, int(AFFORDABILITY_CAP * w.monthly_income * w.tenure_months))
        loans = list(range(low, most + 1, WORKER_LOAN_STEP))
        if not loans:
            return {"approve_up_to": None, "review_up_to": None}
        risks = self._risks([self._row(replace(w, loan_amount=loan)) for loan in loans])

        def limit(cutoff):
            return next((loans[i] for i in range(len(loans) - 1, -1, -1) if risks[i] <= cutoff), None)

        return {"approve_up_to": limit(thresholds["approve"]), "review_up_to": limit(thresholds["high_risk"])}

    def recourse(self, w: Worker) -> list:
        """The smallest change to the loan, or to bill-paying habits, that reaches the approve band."""
        self.validate(w)
        cutoff = self.meta["thresholds"]["approve"]
        if self._risks([self._row(w)])[0] <= cutoff:
            return []
        options = []
        loans = list(range(w.loan_amount - WORKER_LOAN_STEP, WORKER_LOAN_LIMITS[0] - 1, -WORKER_LOAN_STEP))
        if loans:
            risks = self._risks([self._row(replace(w, loan_amount=loan)) for loan in loans])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                options.append({"kind": "smaller_loan", "loan_amount": loans[hits[0]], "risk": float(risks[hits[0]])})
        shares = [
            share / 100
            for share in range(math.ceil(w.bills_on_time * 100 / 5) * 5, 101, 5)
            if share / 100 > w.bills_on_time
        ]
        if shares:
            risks = self._risks([self._row(replace(w, bills_on_time=share)) for share in shares])
            hits = np.flatnonzero(risks <= cutoff)
            if hits.size:
                options.append({"kind": "bills", "bills_on_time": shares[hits[0]], "risk": float(risks[hits[0]])})
        return options
