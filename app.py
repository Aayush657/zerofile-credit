"""ZeroFile Streamlit app. It reads only artifacts/, so no borrower data ever reaches the host."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from zerofile.scoring import (
    AFFORDABILITY_CAP,
    ARTIFACTS,
    EMPLOYMENT_CODES,
    TIER_LABELS,
    Applicant,
    Scorer,
    Worker,
    WorkerScorer,
    net_value,
)

BLUE, ORANGE, INK, SECONDARY, MUTED, GRID, AXIS = (
    "#2a78d6",
    "#eb6834",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
DISCLAIMER = "Research prototype for Neural Storm 2026. Not a lending decision or financial advice. All applicants shown are fictional."
REQUIRED_FILES = (
    "model.txt",
    "meta.json",
    "report.json",
    "worker_model.txt",
    "worker_meta.json",
    "worker_report.json",
    "tuning.json",
)

# Fictional examples, picked to show each band. No row comes from a dataset.
PERSONAS = {
    "Self-employed delivery rider, a close call": dict(
        vehicle_price=68_000,
        down_payment=16_000,
        employment="Self employed",
        brand="average",
        dealer="average",
        branch="average",
    ),
    "Salaried clinic assistant, larger down payment": dict(
        vehicle_price=72_000,
        down_payment=27_000,
        employment="Salaried",
        brand="average",
        dealer="better",
        branch="better",
    ),
    "Self-employed tailor, small down payment": dict(
        vehicle_price=82_000,
        down_payment=8_000,
        employment="Self employed",
        brand="worse",
        dealer="worse",
        branch="worse",
    ),
}
WORKERS = {
    "Rural farm worker, no UPI": dict(
        w_bills=45,
        w_has_rent=False,
        w_rent=50,
        w_recharges=1.5,
        w_uses_upi=False,
        w_upi_payments=30,
        w_upi_months=12,
        w_income=13_000,
        w_swing=35,
        w_loan=60_000,
        w_tenure=24,
    ),
    "Gig delivery rider, steady habits": dict(
        w_bills=65,
        w_has_rent=True,
        w_rent=60,
        w_recharges=1.75,
        w_uses_upi=True,
        w_upi_payments=45,
        w_upi_months=15,
        w_income=24_000,
        w_swing=40,
        w_loan=120_000,
        w_tenure=36,
    ),
    "Seasonal worker, missed bills": dict(
        w_bills=30,
        w_has_rent=False,
        w_rent=50,
        w_recharges=1.25,
        w_uses_upi=False,
        w_upi_payments=30,
        w_upi_months=12,
        w_income=11_000,
        w_swing=50,
        w_loan=100_000,
        w_tenure=18,
    ),
}
HIDDEN_WORKER_INPUTS = ("w_rent", "w_upi_payments", "w_upi_months")
WEEK_STARTS = {40: "1 Oct", 41: "8 Oct", 42: "15 Oct", 43: "22 Oct", 44: "29 Oct"}
AGE_ORDER = ["Under 25", "25 to 34", "35 to 44", "45 and over"]

st.set_page_config(page_title="ZeroFile", page_icon="📄", layout="wide")


@st.cache_resource
def get_models() -> tuple:
    return Scorer(), WorkerScorer()


@st.cache_data
def get_reports() -> tuple:
    return tuple(
        json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
        for name in ("report.json", "worker_report.json", "tuning.json")
    )


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def polish(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID,
            domainColor=AXIS,
            tickColor=AXIS,
            labelColor=SECONDARY,
            titleColor=SECONDARY,
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(labelColor=SECONDARY, labelFontSize=12)
        .configure(background="transparent")
    )


def show(chart) -> None:
    st.altair_chart(polish(chart), width="stretch", theme=None)


def real_badge() -> None:
    st.badge("Real repayment outcomes", icon=":material/verified:", color="green")


def synthetic_badge() -> None:
    st.badge("Synthetic training data", icon=":material/science:", color="orange")


def compare_lines(
    wide: pd.DataFrame,
    x: str,
    x_title: str,
    series: dict,
    y_title: str,
    kind: str = "Q",
    x_format: str | None = None,
    extra: list | None = None,
):
    """Two series on one axis, with a crosshair and a tooltip showing both values."""
    long = wide.melt(id_vars=[x], value_vars=list(series), var_name="column", value_name="value")
    long["Series"] = long["column"].map(series)
    axis = alt.Axis(labelAngle=0, format=x_format, tickCount=10) if x_format else alt.Axis(labelAngle=0)
    x_encoding = alt.X(f"{x}:{kind}", title=x_title, axis=axis, sort=None)
    color = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=list(series.values()), range=[BLUE, ORANGE]),
        legend=alt.Legend(orient="top", title=None),
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=x_encoding,
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
            color=color,
        )
    )
    nearest = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    x_tip = (
        alt.Tooltip(f"{x}:{kind}", title=x_title, format=x_format)
        if x_format
        else alt.Tooltip(f"{x}:{kind}", title=x_title)
    )
    hover = (
        alt.Chart(wide)
        .mark_rule(color=MUTED, strokeWidth=1)
        .encode(
            x=x_encoding,
            opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
            tooltip=[x_tip] + [alt.Tooltip(f"{column}:Q", title=name, format=".1%") for column, name in series.items()],
        )
        .add_params(nearest)
    )
    dots = lines.mark_point(filled=True, size=70, opacity=1).transform_filter(nearest)
    return alt.layer(lines, *(extra or []), dots, hover).properties(height=300)


def score_scale(points: float, high_at: float, approve_at: float):
    bands = pd.DataFrame(
        [
            {"start": 300, "end": high_at, "band": "High risk"},
            {"start": high_at, "end": approve_at, "band": "Human review"},
            {"start": approve_at, "end": 900, "band": "Approve"},
        ]
    )
    bands["middle"] = (bands["start"] + bands["end"]) / 2
    x = alt.X(
        "start:Q",
        scale=alt.Scale(domain=[300, 900]),
        axis=alt.Axis(
            title=None, values=[300, round(high_at), round(approve_at), 900], grid=False, domain=False, ticks=False
        ),
    )
    bar = (
        alt.Chart(bands)
        .mark_bar(height=10, cornerRadius=2)
        .encode(
            x=x,
            x2="end:Q",
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=["High risk", "Human review", "Approve"], range=[CRITICAL, WARNING, GOOD]),
                legend=None,
            ),
            tooltip=["band:N"],
        )
    )
    names = (
        alt.Chart(bands)
        .mark_text(dy=-20, fontSize=12, color=SECONDARY)
        .encode(x=alt.X("middle:Q", scale=alt.Scale(domain=[300, 900])), text="band:N")
    )
    marker = (
        alt.Chart(pd.DataFrame({"points": [points]}))
        .mark_tick(color=INK, thickness=3, size=20)
        .encode(x=alt.X("points:Q", scale=alt.Scale(domain=[300, 900])))
    )
    return alt.layer(bar, names, marker).properties(height=96)


def breakdown_chart(reasons: list):
    rows = pd.DataFrame(reasons)
    rows["label"] = rows["points"].map(lambda value: f"{value:+.0f}")
    y = alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=160))
    x = alt.X(
        "points:Q",
        title="Points gained or lost against the average applicant",
        axis=alt.Axis(tickCount=6),
        scale=alt.Scale(padding=40),
    )
    tip = [
        alt.Tooltip("factor:N", title="Factor"),
        alt.Tooltip("label:N", title="Points"),
        alt.Tooltip("detail:N", title="Why"),
    ]
    bars = (
        alt.Chart(rows)
        .mark_bar(size=16, cornerRadius=3)
        .encode(y=y, x=x, color=alt.condition(alt.datum.points >= 0, alt.value(BLUE), alt.value(ORANGE)), tooltip=tip)
    )
    gains = (
        alt.Chart(rows[rows["points"] >= 0])
        .mark_text(align="left", dx=5, color=SECONDARY)
        .encode(y=y, x="points:Q", text="label:N")
    )
    losses = (
        alt.Chart(rows[rows["points"] < 0])
        .mark_text(align="right", dx=-5, color=SECONDARY)
        .encode(y=y, x="points:Q", text="label:N")
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return alt.layer(bars, zero, gains, losses).properties(height=44 * len(rows) + 40)


def band_box(band: str, message: str) -> None:
    if band == "Approve":
        st.success(f"**Approve.** {message}", icon="✅")
    elif band == "Human review":
        st.warning(f"**Human review.** {message}", icon="⚠️")
    else:
        st.error(f"**High risk.** A person should review this before any decline. {message}", icon="⛔")


def loan_suggestion(loans: dict) -> None:
    if loans["approve_up_to"]:
        st.markdown(f"**Up to ₹{loans['approve_up_to']:,}** stays in the approve band.")
    else:
        st.markdown("No loan amount here reaches the approve band.")
    if loans["review_up_to"] and loans["review_up_to"] != loans["approve_up_to"]:
        st.markdown(
            f"Up to ₹{loans['review_up_to']:,} goes to human review. Anything larger is high risk or unaffordable."
        )


def why_this_score(result) -> None:
    st.subheader("Why this score")
    st.write(
        f"SHAP values show what each factor did. The average applicant starts at {result.base_points:.0f} points, and these factors add {sum(result.factors.values()):+.0f}, which gives {result.points:.0f}."
    )
    show(breakdown_chart(result.reasons))
    for reason in result.reasons:
        st.markdown(f"- **{reason['factor']} ({reason['points']:+.0f}):** {reason['detail']}")


def load_persona() -> None:
    st.session_state.update(PERSONAS[st.session_state.persona])


def load_worker() -> None:
    st.session_state.update(WORKERS[st.session_state.worker_example])


def worker_tab(scorer: WorkerScorer, worker_report: dict) -> None:
    synthetic_badge()
    st.write(
        "This is what the HI-04 brief asks for: a score from bills, mobile and income, with no credit history. No public dataset links these signals to repayment, so this model learned from an open synthetic dataset. Treat it as a working demonstration, not proof."
    )
    if "w_bills" not in st.session_state:
        st.session_state.update(next(iter(WORKERS.values())))
    form, output = st.columns([1, 2], gap="large")
    with form:
        st.selectbox("Start from an example", WORKERS, key="worker_example", on_change=load_worker)
        st.slider("Utility bills paid on time (%)", 0, 100, step=5, key="w_bills")
        st.toggle("Has a rent record", key="w_has_rent")
        example = WORKERS[st.session_state.worker_example]
        for name in HIDDEN_WORKER_INPUTS:
            # Streamlit forgets a widget's value while it is hidden.
            if name not in st.session_state:
                st.session_state[name] = example[name]
        if st.session_state.w_has_rent:
            st.slider("Rent paid on time (%)", 0, 100, step=5, key="w_rent")
        st.slider("Mobile recharges a month", 0.5, 4.0, step=0.25, key="w_recharges")
        st.toggle("Uses UPI", key="w_uses_upi")
        if st.session_state.w_uses_upi:
            st.slider("UPI payments a month", 0, 120, step=5, key="w_upi_payments")
            st.slider("Months active on UPI", 0, 36, key="w_upi_months")
        st.number_input("Average monthly income (₹)", min_value=1_000, max_value=500_000, step=500, key="w_income")
        st.slider("How much income swings month to month (%)", 0, 80, step=5, key="w_swing")
        st.number_input("Loan requested (₹)", min_value=1_000, max_value=2_000_000, step=1_000, key="w_loan")
        st.slider("Repayment period (months)", 6, 60, step=6, key="w_tenure")

    state = st.session_state
    worker = Worker(
        bills_on_time=state.w_bills / 100,
        recharges_per_month=state.w_recharges,
        monthly_income=state.w_income,
        income_swing=state.w_swing / 100,
        loan_amount=state.w_loan,
        tenure_months=state.w_tenure,
        rent_on_time=state.w_rent / 100 if state.w_has_rent else None,
        upi_transactions=state.w_upi_payments if state.w_uses_upi else None,
        upi_months=state.w_upi_months if state.w_uses_upi else None,
    )
    with output:
        try:
            result = scorer.score(worker)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
            return
        thresholds = scorer.meta["thresholds"]
        first, second, third = st.columns(3)
        first.metric("ZeroFile Score", f"{result.points:.0f}", help="300 to 900. This is not a CIBIL score.")
        second.metric("Estimated risk of default", pct(result.risk))
        third.metric("Instalment against income", pct(worker.emi_to_income, 0), help="Before interest.")
        show(
            score_scale(
                result.points,
                scorer.points_for_risk(thresholds["high_risk"]),
                scorer.points_for_risk(thresholds["approve"]),
            )
        )
        band_default = {band["band"]: band["default"] for band in worker_report["bands"]}[result.band]
        band_box(
            result.band, f"In the synthetic test set, workers in this band defaulted {pct(band_default)} of the time."
        )
        for note in result.warnings:
            st.info(note, icon="ℹ️")

        st.subheader("Suggested loan")
        loan_suggestion(scorer.loan_range(worker))
        st.caption(
            f"Over {state.w_tenure} months, and capped so the instalment stays under {AFFORDABILITY_CAP:.0%} of income. The cap is a policy rule, not something the model learned."
        )

        if result.band != "Approve":
            st.subheader("What would change this result")
            options = scorer.recourse(worker)
            if not options:
                st.write(
                    "No smaller loan or better bill record alone reaches the approve band, so this needs a person."
                )
            for option in options:
                if option["kind"] == "smaller_loan":
                    st.markdown(
                        f"**Borrow ₹{option['loan_amount']:,} instead.** Estimated risk falls to {pct(option['risk'])}, which reaches the approve band."
                    )
                else:
                    st.markdown(
                        f"**Pay {option['bills_on_time']:.0%} of utility bills on time.** Estimated risk falls to {pct(option['risk'])}."
                    )
            if options:
                st.caption("Model estimates from synthetic data, not a promise of approval.")
        why_this_score(result)


def applicant_tab(scorer: Scorer, report: dict) -> None:
    real_badge()
    st.write("This model learned from real vehicle loans. It uses no bureau data, ID documents, age or location.")
    if "vehicle_price" not in st.session_state:
        st.session_state.update(next(iter(PERSONAS.values())))

    def tier_label(tier):
        return TIER_LABELS[tier].capitalize()

    form, output = st.columns([1, 2], gap="large")
    with form:
        st.selectbox("Start from an example", PERSONAS, key="persona", on_change=load_persona)
        st.number_input("Vehicle price (₹)", min_value=10_000, max_value=5_000_000, step=1_000, key="vehicle_price")
        st.number_input("Down payment (₹)", min_value=0, max_value=5_000_000, step=500, key="down_payment")
        st.selectbox("Employment type", list(EMPLOYMENT_CODES), key="employment")
        st.selectbox("Vehicle brand's repayment record", TIER_LABELS, format_func=tier_label, key="brand")
        st.selectbox("Dealer's repayment record", TIER_LABELS, format_func=tier_label, key="dealer")
        st.selectbox("Branch's repayment record", TIER_LABELS, format_func=tier_label, key="branch")
        st.caption(
            "A live system would look these records up from the dealer and branch. They are choices here because the dataset's IDs are anonymous."
        )

    applicant = Applicant(
        **{
            field: st.session_state[field]
            for field in ("vehicle_price", "down_payment", "employment", "brand", "dealer", "branch")
        }
    )
    with output:
        try:
            result = scorer.score(applicant)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
            return
        thresholds = scorer.meta["thresholds"]
        first, second, third = st.columns(3)
        first.metric(
            "ZeroFile Score",
            f"{result.points:.0f}",
            help="300 to 900. Every 100 points halve the odds of default. This is not a CIBIL score.",
        )
        second.metric("Estimated risk of default", pct(result.risk))
        third.metric("Loan against vehicle price", f"{applicant.ltv_pct:.0f}%")
        show(
            score_scale(
                result.points,
                scorer.points_for_risk(thresholds["high_risk"]),
                scorer.points_for_risk(thresholds["approve"]),
            )
        )
        band_default = {band["band"]: band["default"] for band in report["bands"]}[result.band]
        band_box(
            result.band,
            f"In the October back-test, no-file applicants in this band defaulted {pct(band_default)} of the time.",
        )
        for note in result.warnings:
            st.info(note, icon="ℹ️")

        st.subheader("Suggested loan on this vehicle")
        loan_suggestion(scorer.loan_range(applicant))

        options = scorer.recourse(applicant)
        if result.band != "Approve":
            st.subheader("What would change this result")
            if not options:
                st.write(
                    "No change to the down payment or the vehicle alone reaches the approve band, so this needs a person."
                )
            for option in options:
                if option["kind"] == "down_payment":
                    st.markdown(
                        f"**Put ₹{option['extra']:,} more down** (₹{option['down_payment']:,} in total). Estimated risk falls to {pct(option['risk'])}, which reaches the approve band."
                    )
                else:
                    st.markdown(
                        f"**Or choose a vehicle around ₹{option['vehicle_price']:,}** with the same down payment. Estimated risk falls to {pct(option['risk'])}."
                    )
            if options:
                st.caption(
                    "Estimates from past loans, not a promise of approval. Identity, dealer and age are never suggested as changes."
                )
        why_this_score(result)


def lender_tab(report: dict) -> None:
    headline = report["headline"]
    real_badge()
    st.write(
        f"Every applicant here has no credit file, so a lender using only a bureau score approves none of them. "
        f"Move the slider to choose how many ZeroFile approves. All figures come from {headline['applicants']:,} October loans the model never trained on."
    )
    gig = next(row for row in report["fairness"]["Employment"] if row["group"] == "Self employed")
    st.write(
        f"**Closest group to gig workers in this data:** {gig['n']:,} self-employed applicants, of whom {pct(gig['approval'])} are approved and {pct(gig['default_if_approved'])} of those default. "
        "The data does not mark rural applicants or specific gig jobs."
    )
    curve = pd.DataFrame(report["policy_curve"])
    share = st.slider("Approve the safest share of no-file applicants", 10, 100, 42, step=2, format="%d%%", key="share")
    index = (curve["share"] - share / 100).abs().argmin()
    row = curve.iloc[index]
    first, second, third, fourth = st.columns(4)
    first.metric("People approved", f"{row['people']:,.0f}")
    second.metric("Credit unlocked", f"₹{row['crore']:,.0f} crore")
    third.metric(
        "Their default rate",
        pct(row["default"]),
        f"{100 * (row['default'] - headline['benchmark_default']):+.1f} pts against bureau low-risk",
        delta_color="inverse",
    )
    fourth.metric(
        "LTV rule at the same size",
        pct(row["ltv_rule_default"]),
        f"{100 * (row['ltv_rule_default'] - row['default']):+.1f} pts worse than ZeroFile",
        delta_color="off",
    )

    wide = curve.rename(columns={"default": "zerofile", "ltv_rule_default": "ltv_rule"})
    benchmark = alt.Chart(
        pd.DataFrame(
            {
                "y": [headline["benchmark_default"]],
                "label": [f"Bureau low-risk borrowers: {pct(headline['benchmark_default'])}"],
                "x": [wide["share"].min()],
            }
        )
    )
    extra = [
        benchmark.mark_rule(color=SECONDARY, strokeWidth=1).encode(y="y:Q"),
        benchmark.mark_text(align="left", dy=-8, color=SECONDARY, fontSize=12).encode(x="x:Q", y="y:Q", text="label:N"),
        alt.Chart(pd.DataFrame([row]))
        .mark_point(filled=True, size=140, color=BLUE, stroke="#fcfcfb", strokeWidth=2, opacity=1)
        .encode(x="share:Q", y="default:Q"),
    ]
    show(
        compare_lines(
            wide,
            "share",
            "Share of no-file applicants approved",
            {"zerofile": "ZeroFile", "ltv_rule": "Simple LTV rule"},
            "Default rate among those approved",
            x_format=".0%",
            extra=extra,
        )
    )
    st.caption(
        f"The slider shows the whole range after the fact. The headline result is stricter: its cutoff was fixed on September data, before October was scored, and approves {pct(headline['share'])} of applicants at a {pct(headline['default'])} default rate. "
        "Every row is a loan the lender actually funded, so this is a back-test."
    )
    with st.expander("See these numbers as a table"):
        table = pd.DataFrame(
            {
                "Approved": curve["share"].map(lambda v: pct(v, 0)),
                "People": curve["people"],
                "Credit (₹ crore)": curve["crore"].round(1),
                "ZeroFile default rate": curve["default"].map(pct),
                "LTV rule default rate": curve["ltv_rule_default"].map(pct),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("What is that worth to the lender?")
    st.write(
        "Enter your own economics. These two numbers are your assumptions, not data, and the result covers one month of loans."
    )
    left, right = st.columns(2)
    margin = (
        left.number_input("Margin earned on a loan that repays (%)", min_value=1, max_value=50, value=12, key="margin")
        / 100
    )
    loss = (
        right.number_input(
            "Share of a loan lost when it defaults (%)", min_value=1, max_value=100, value=50, key="loss"
        )
        / 100
    )
    values = net_value(curve["crore"], curve["default"], margin, loss)
    best = int(values.argmax())
    first, second = st.columns(2)
    first.metric(f"Net value at {pct(row['share'], 0)} approved", f"₹{values.iloc[index]:,.1f} crore")
    second.metric(
        "Best approval level for these numbers",
        pct(curve["share"].iloc[best], 0),
        f"₹{values.iloc[best]:,.1f} crore",
        delta_color="off",
    )

    st.subheader("What would utility and mobile data add to these real loans?")
    st.warning(
        "**This is a simulation, not a result.** It adds generated utility, mobile and income signals to the real loans at an assumed strength, to show what a lender could gain if real signals were this strong. All other numbers on this page are measured.",
        icon="⚠️",
    )
    scenarios = report["alt_scenarios"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Scenario": "Today, without these signals (measured)",
                    "Assumed AUC of each signal": "None",
                    "Model AUC": f"{report['auc']['no_bureau']['value']:.3f}",
                    "Approved": pct(headline["share"]),
                    "People": f"{headline['people']:,}",
                    "Credit (₹ crore)": f"{headline['crore']:.0f}",
                    "Default rate": pct(headline["default"]),
                },
                *[
                    {
                        "Scenario": f"{row_['scenario']} (simulated)",
                        "Assumed AUC of each signal": f"{row_['signal_auc']:.2f}",
                        "Model AUC": f"{row_['auc_no_bureau']:.3f}",
                        "Approved": pct(row_["share"]),
                        "People": f"{row_['people']:,}",
                        "Credit (₹ crore)": f"{row_['crore']:.0f}",
                        "Default rate": pct(row_["default"]),
                    }
                    for row_ in scenarios
                ],
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    moderate = next(row_ for row_ in scenarios if row_["scenario"] == "Moderate")
    st.caption(
        f"The Moderate signals separate the riskiest and safest fifths of applicants by {moderate['signals_only_quintile_ratio']:.1f} times. For comparison, [one study of real mobile phone records](https://arxiv.org/abs/1712.05840v3) reported 2.8 times on a different population. "
        "Real signals would overlap with the inputs already used, so real gains may be smaller."
    )


def evidence_tab(report: dict, worker_report: dict, tuning: dict) -> None:
    auc, periods = report["auc"], report["periods"]
    no_bureau, gain = auc["no_bureau"], auc["rated_gain"]
    classification = report["headline"]["classification"]
    st.subheader("Vehicle-loan model")
    real_badge()
    first, second, third, fourth = st.columns(4)
    first.metric(
        "AUC, no-file applicants",
        f"{no_bureau['value']:.3f}",
        help=f"95% bootstrap interval {no_bureau['low']:.3f} to {no_bureau['high']:.3f}. A bureau score cannot rank these people at all, because they have none.",
    )
    second.metric("KS statistic", f"{report['ks_no_bureau']:.3f}")
    third.metric(
        "Brier score",
        f"{report['brier_no_bureau']:.3f}",
        f"Base rate alone: {report['brier_base_rate_no_bureau']:.3f}",
        delta_color="off",
    )
    fourth.metric(
        "Scoring time",
        f"{report['latency_ms_p50']:.1f} ms",
        help="Median for one applicant, including the points breakdown.",
    )
    st.caption(
        f"Trained on {periods['train']['rows']:,} loans ({periods['train']['start']} to {periods['train']['end']}), tuned on {periods['validation']['rows']:,} ({periods['validation']['start']} to {periods['validation']['end']}), "
        f"and tested on {periods['test']['rows']:,} from {periods['test']['start']} to {periods['test']['end']}. The split is by date because default rates drift by month."
    )

    st.markdown("##### It out-ranks the bureau score, even where a score exists")
    st.write(
        f"On borrowers with a real bureau grade, ZeroFile reaches AUC {auc['rated_zerofile']['value']:.3f} against {auc['rated_bureau_score']['value']:.3f} for the bureau score itself. "
        f"The gain is {gain['value']:.3f} (95% CI {gain['low']:.3f} to {gain['high']:.3f}). Full bureau data is still valuable where it exists, as the last row below shows, so ZeroFile is a way in for people without a file and not a replacement for the bureau."
    )
    variants = pd.DataFrame(report["variants"])
    st.dataframe(
        pd.DataFrame(
            {
                "Model": variants["name"],
                "AUC, no-file applicants": variants["auc_no_bureau"].map("{:.3f}".format),
                "AUC, bureau-rated borrowers": variants["auc_rated"].map("{:.3f}".format),
                "Lowest regional approval, as a share of the highest": variants["worst_region_ratio"].map(
                    lambda v: pct(v, 0)
                ),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        f"**Is 0.631 the best this data can do?** Very likely. A plain logistic regression on the same inputs reaches {report['logistic_baseline']['auc_no_bureau']:.3f}, "
        f"and a scan of {tuning['configurations']} LightGBM settings only moved it between {tuning['auc_test_low']:.3f} and {tuning['auc_test_high']:.3f}. First-instalment default is a noisy target."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Risk ranking and calibration")
        rows = pd.DataFrame(report["calibration"])
        rows["decile"] = range(1, len(rows) + 1)
        show(
            compare_lines(
                rows,
                "decile",
                "Risk decile, safest to riskiest",
                {"actual": "Actual default rate", "predicted": "Predicted"},
                "Default rate",
                kind="O",
            )
        )
        st.caption(
            f"The riskiest tenth defaults {rows['actual'].iloc[-1] / rows['actual'].iloc[0]:.1f} times as often as the safest. Expected calibration error: {pct(report['ece_no_bureau'])}."
        )
    with right:
        st.markdown("##### Predictions ran low as October went on")
        weeks = pd.DataFrame(report["weekly_drift"])
        weeks["week"] = weeks["week"].map(WEEK_STARTS)
        show(
            compare_lines(
                weeks,
                "week",
                "Week starting",
                {"actual": "Actual default rate", "predicted": "Predicted"},
                "Default rate",
                kind="N",
            )
        )
        st.caption(
            "Defaults jumped at the end of October, and the model did not see that coming. Ranking held up better than absolute risk, so a live system would need regular recalibration."
        )

    st.markdown("##### What moves the score most")
    importance = pd.DataFrame(report["importance"])
    bars = (
        alt.Chart(importance)
        .mark_bar(size=16, cornerRadiusEnd=4, color=BLUE)
        .encode(
            y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False)),
            x=alt.X("points:Q", title="Average effect on the score, in points", axis=alt.Axis(tickCount=6)),
            tooltip=[
                alt.Tooltip("factor:N", title="Factor"),
                alt.Tooltip("points:Q", title="Average points", format=".1f"),
            ],
        )
    )
    show(bars.properties(height=44 * len(importance) + 40))
    st.caption(
        f"Treating 'not approved' as a prediction of default gives recall {pct(classification['recall'], 0)}, precision {pct(classification['precision'], 0)}, F1 {classification['f1']:.2f} and accuracy {pct(classification['accuracy'], 0)}. "
        "The lender picks this trade-off with the slider, so ranking and default rate matter more here than a single accuracy figure."
    )

    st.divider()
    st.subheader("Worker model")
    synthetic_badge()
    worker_auc = worker_report["auc"]
    st.warning(
        "**These numbers show the pipeline works. They do not show real-world accuracy.** In this synthetic dataset, a borrower is labelled as defaulted whenever a generated probability passes a fixed cutoff, so a model can learn the generator's rules. The real figure to compare against is the 0.631 above.",
        icon="⚠️",
    )
    first, second, third, fourth = st.columns(4)
    first.metric(
        "AUC, test set",
        f"{worker_auc['value']:.3f}",
        help=f"95% bootstrap interval {worker_auc['low']:.3f} to {worker_auc['high']:.3f}.",
    )
    second.metric(
        "Across 5 random splits",
        f"{min(worker_report['auc_by_split']):.3f} to {max(worker_report['auc_by_split']):.3f}",
    )
    third.metric("Logistic regression, same inputs", f"{worker_report['auc_logistic']:.3f}")
    fourth.metric("Calibration error", pct(worker_report["ece"]))
    worker_variants = pd.DataFrame(worker_report["variants"])
    st.dataframe(
        pd.DataFrame(
            {
                "Inputs": worker_variants["name"],
                "AUC": worker_variants["auc"].map("{:.3f}".format),
                "Lowest worker-type share not marked high risk, against the highest": worker_variants[
                    "worst_type_ratio"
                ].map(lambda v: pct(v, 0)),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Wealth stand-ins, surveys, e-commerce and demographics add nothing here, so they stay out of the model."
    )


def audit_view(rows: list, name: str, extra: list | None = None) -> None:
    """A bar for each group's approval, its flag, and the same numbers as a table."""
    frame = pd.DataFrame(rows)
    if name == "Age band":
        frame = frame.set_index("group").loc[AGE_ORDER].reset_index()
    elif name in ("Region", "State"):
        frame = frame.sort_values("approval", ascending=False)
        if name == "Region":
            frame["group"] = "Region " + frame["group"]
    frame["status"] = frame["passes_four_fifths"].map({True: "Passes", False: "Below four-fifths"})
    line = frame["approval"].max() * 0.8
    y = alt.Y("group:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=160))
    bars = (
        alt.Chart(frame)
        .mark_bar(size=14, cornerRadiusEnd=4)
        .encode(
            y=y,
            x=alt.X("approval:Q", title="Share reaching this outcome", axis=alt.Axis(format=".0%", tickCount=8)),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["Passes", "Below four-fifths"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("approval:Q", title="Share", format=".1%"),
                alt.Tooltip("default:Q", title="Actual default rate", format=".1%"),
                alt.Tooltip("status:N", title="Four-fifths rule"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(color=SECONDARY, strokeWidth=1).encode(x="x:Q")
    show(alt.layer(bars, rule).properties(height=32 * len(frame) + 70))
    st.caption(f"The vertical line marks four-fifths of the best group's share ({pct(line)}).")
    table = {
        "Group": frame["group"],
        "Applicants": frame["n"],
        "Share reaching this outcome": frame["approval"].map(pct),
        "Actual default rate": frame["default"].map(pct),
        "Default rate within that share": frame["default_if_approved"].map(pct),
        "Share among those who repaid": frame["approval_if_repaid"].map(pct),
        "Four-fifths rule": frame["status"],
    }
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def real_fairness(report: dict) -> None:
    fairness, floor, headline = report["fairness"], report["regional_floor"], report["headline"]
    regions = fairness["Region"]
    failing = [row for row in regions if not row["passes_four_fifths"]]
    real_badge()
    st.write(
        "Age, region and ID documents are **not model inputs**. This audit checks the result anyway, because other inputs can stand in for them. "
        "The test is the four-fifths rule: a group is flagged if its approval rate is under 80% of the best group's. "
        "**Gender is not in this dataset, so it could not be audited.**"
    )
    first, second, third = st.columns(3)
    for column, name in ((first, "Age band"), (second, "Employment")):
        rows = fairness[name]
        column.metric(name, f"{sum(row['passes_four_fifths'] for row in rows)} of {len(rows)} pass")
    third.metric("Region", f"{len(regions) - len(failing)} of {len(regions)} pass")

    worst = min(regions, key=lambda row: row["approval"])
    overlooked = min(failing, key=lambda row: row["default"])
    missed = max(regions, key=lambda row: row["default_if_approved"] or 0)
    st.warning(
        f"**Region fails, and this needs a person in the loop.** Region {worst['group']} gets {pct(worst['approval'])} approval with a {pct(worst['default'])} default rate, so the direction is right but the gap is far wider than the risk gap. "
        f"Region {overlooked['group']} defaults at {pct(overlooked['default'])}, below average, yet only {pct(overlooked['approval'])} are approved. "
        f"In Region {missed['group']}, {pct(missed['approval'])} are approved and {pct(missed['default_if_approved'])} of those default, so the model is missing something there.",
        icon="⚠️",
    )

    st.markdown("##### A measured fix: a regional approval floor")
    st.write(
        f"Require every region to approve at least {pct(floor['floor'])} of its no-file applicants (80% of the overall rate), safest applicants first."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Policy": ["Model cutoff alone", "With the regional floor"],
                "People approved": [f"{headline['people']:,}", f"{floor['people']:,}"],
                "Credit (₹ crore)": [f"{headline['crore']:.0f}", f"{floor['crore']:.0f}"],
                "Default rate": [pct(headline["default"]), pct(floor["default"])],
                "Regions under the floor": [floor["regions_below_before"], floor["regions_below_after"]],
                "Lowest regional approval, as a share of the highest": [
                    pct(min(row["ratio"] for row in regions), 0),
                    pct(floor["worst_ratio"], 0),
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"The floor brings in {floor['people_added']:,} more people and stays below the {pct(headline['benchmark_default'])} bureau low-risk benchmark. It narrows the regional gap but does not close it, and it was computed on October after the fact."
    )

    st.markdown("##### Full audit")
    name = st.radio("Group by", list(fairness), horizontal=True, key="audit_dimension")
    audit_view(fairness[name], name)


def worker_fairness(worker_report: dict) -> None:
    outright, reach = worker_report["fairness"], worker_report["fairness_not_high_risk"]
    synthetic_badge()
    st.write(
        "Worker type, state and employment are **not model inputs**, and neither is income level. Each group is tested twice: **approved outright**, and **not marked high risk**, which is everyone who reaches a yes or a person. "
        "**Gender is not in this dataset, so it could not be audited.**"
    )
    first, second, third, fourth = st.columns(4)
    for column, name in zip((first, second, third, fourth), reach, strict=False):
        column.metric(
            name,
            f"{sum(row['passes_four_fifths'] for row in reach[name])} of {len(reach[name])} pass",
            help="Using the 'not marked high risk' test.",
        )

    def find(rows, group):
        return next(row for row in rows if row["group"] == group)

    rural_outright, rural_reach = find(outright["Worker type"], "rural"), find(reach["Worker type"], "rural")
    no_upi_outright, no_upi_reach = (
        find(outright["UPI access"], "No UPI history"),
        find(reach["UPI access"], "No UPI history"),
    )
    with_upi_reach = find(reach["UPI access"], "Uses UPI")
    rural_smaller = find(worker_report["smaller_loan"]["by_type"], "rural")
    st.warning(
        f"**Rural workers and people without UPI do worse, and a person should see their cases.** Only {pct(rural_outright['approval'], 0)} of rural workers are approved outright, which fails the rule, though {pct(rural_reach['approval'], 0)} reach a yes or a person, which passes. "
        f"Without UPI history, {pct(no_upi_outright['approval'], 0)} are approved outright and {pct(no_upi_reach['approval'], 0)} reach a yes or a person, against {pct(with_upi_reach['approval'], 0)} of UPI users, which fails. "
        f"The synthetic data has them defaulting at {pct(no_upi_reach['default'], 0)} against {pct(with_upi_reach['default'], 0)}. That may be the dataset author's assumption and not how real people behave, so the app flags every no-UPI result for review.",
        icon="⚠️",
    )
    st.caption(
        f"Offering a smaller loan would lift rural approval from {pct(rural_smaller['approved'], 0)} to {pct(rural_smaller['with_smaller_loan'], 0)}. But {pct(worker_report['smaller_loan']['rescued_default_on_loan_asked'], 0)} of those people defaulted on the larger loan they asked for, and this data cannot say whether a smaller one would have helped, so this advice is untested."
    )
    name = st.radio("Group by", list(reach), horizontal=True, key="worker_audit_dimension")
    outcome = st.radio(
        "Outcome", ["Not marked high risk", "Approved outright"], horizontal=True, key="worker_audit_outcome"
    )
    audit_view((reach if outcome == "Not marked high risk" else outright)[name], name)


def fairness_tab(report: dict, worker_report: dict) -> None:
    model = st.radio("Which model?", ["Vehicle-loan model", "Worker model"], horizontal=True, key="fairness_model")
    if model == "Vehicle-loan model":
        real_fairness(report)
    else:
        worker_fairness(worker_report)


def limits_tab() -> None:
    st.subheader("What this is")
    st.write(
        "A decision aid and research prototype. It should send borderline cases to a person, and it should never decline someone on its own."
    )
    st.subheader("Two models, kept apart")
    st.markdown(
        """
- **Vehicle-loan model:** learned from real repayment outcomes. Every number across the top of this page comes from it.
- **Worker model:** learned from synthetic data, because no public dataset links utility or mobile records to repayment. It shows how such a score would work, and proves nothing about real accuracy.
        """
    )
    st.subheader("Known limits")
    st.markdown(
        """
- **Synthetic labels.** In the worker data, default is a generated probability above a fixed cutoff, so its high accuracy mostly shows the model recovering those rules. Its risk gaps by region and UPI access are the dataset author's assumptions.
- **It is a back-test.** Every row in the vehicle-loan data is a loan the lender funded. Nobody knows how rejected applicants would have behaved, so results may not carry over to them.
- **Thin-file, not fully unbanked.** Those are 2018 vehicle-loan borrowers with no bureau record. Most probably have bank accounts. That data does not mark rural applicants or gig jobs, so self-employed applicants are the closest stand-in.
- **Modest real signal.** First-instalment default is noisy, and real AUC stays near 0.63. The model separates groups well, but it cannot predict any single person.
- **Risk estimates ran low.** October's defaults rose beyond what the model expected. The bands show what actually happened in October for that reason.
- **Fairness gaps remain.** Region fails on real data, rural and no-UPI groups fare worse on synthetic data, and gender could not be tested in either.
- **Not a fully blind test.** October was inspected while choosing features. Those choices only removed inputs, which should not inflate results, but a truly untouched test does not exist here.
- **Not causal.** A bigger down payment or better bill record lowers the model's estimate. It does not prove that person will repay.
        """
    )
    st.subheader("Privacy and data")
    st.markdown(
        """
- Date of birth, pincode, customer ID and employee code are dropped as the real data loads.
- The deployed app holds models and aggregate results only. It stores nothing you type, and the examples are fictional.
- A live system should use utility, mobile and UPI records only with the applicant's consent.
- Real data: [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction). Its licence is unclear, so raw data is not redistributed.
- Synthetic data: [Alternate Credit Dataset](https://www.kaggle.com/datasets/shreyyans/alternate-credit-dataset), MIT licence.
        """
    )


missing = [name for name in REQUIRED_FILES if not (ARTIFACTS / name).exists()]
if missing:
    st.error(
        f"Missing model files: {', '.join(missing)}. Run `python -m zerofile.data`, then `python -m zerofile.train`, `python -m zerofile.worker` and `python -m zerofile.tune`.",
        icon="🚫",
    )
    st.stop()

scorer, worker_scorer = get_models()
report, worker_report, tuning = get_reports()
headline, population = report["headline"], report["population"]

st.title("ZeroFile")
st.markdown(
    "**A credit score for people with no credit file.** A bureau score cannot rank someone it has never seen. ZeroFile scores them without one, explains every result, suggests a loan they can afford, and shows what would change the answer."
)
st.caption("Measured on real repayment outcomes, on a month the model never trained on:")
first, second, third, fourth = st.columns(4)
first.metric("Borrowers with no credit file", pct(population["no_bureau_share"]))
first.caption(f"{population['no_bureau']:,} of {population['rows']:,} loans in a public Indian vehicle-loan dataset")
second.metric("Approved in the October back-test", f"{headline['people']:,}")
second.caption(f"{pct(headline['share'])} of no-file applicants. A score-only lender approves none.")
third.metric("Credit unlocked in one month", f"₹{headline['crore']:.0f} crore")
third.caption("Cutoff fixed on September data, before October was scored")
fourth.metric("Their default rate", pct(headline["default"]))
fourth.caption(
    f"Bureau low-risk borrowers: {pct(headline['benchmark_default'])}. All no-file applicants: {pct(headline['no_bureau_default'])}."
)

worker_view, applicant_view, lender_view, evidence_view, fairness_view, limits_view = st.tabs(
    ["Score a worker", "Score a vehicle loan", "Lender impact", "Evidence", "Fairness audit", "Limits and data"]
)
with worker_view:
    worker_tab(worker_scorer, worker_report)
with applicant_view:
    applicant_tab(scorer, report)
with lender_view:
    lender_tab(report)
with evidence_view:
    evidence_tab(report, worker_report, tuning)
with fairness_view:
    fairness_tab(report, worker_report)
with limits_view:
    limits_tab()

st.divider()
st.caption(DISCLAIMER)
