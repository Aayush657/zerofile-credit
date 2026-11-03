"""ZeroFile Streamlit app. It reads only artifacts/, so no borrower data ever reaches the host."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from zerofile.scoring import ARTIFACTS, EMPLOYMENT_CODES, TIER_LABELS, Applicant, Scorer

BLUE, ORANGE, INK, SECONDARY, MUTED, GRID, AXIS = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
DISCLAIMER = "Research prototype for Neural Storm 2026. Not a lending decision or financial advice. All applicants shown are fictional."

# Fictional examples, picked to show each band. No row comes from the dataset.
PERSONAS = {
    "Self-employed delivery rider, a close call": dict(vehicle_price=68_000, down_payment=16_000, employment="Self employed", brand="average", dealer="average", branch="average"),
    "Salaried clinic assistant, larger down payment": dict(vehicle_price=72_000, down_payment=27_000, employment="Salaried", brand="average", dealer="better", branch="better"),
    "Self-employed tailor, small down payment": dict(vehicle_price=82_000, down_payment=8_000, employment="Self employed", brand="worse", dealer="worse", branch="worse"),
}
WEEK_STARTS = {40: "1 Oct", 41: "8 Oct", 42: "15 Oct", 43: "22 Oct", 44: "29 Oct"}
AGE_ORDER = ["Under 25", "25 to 34", "35 to 44", "45 and over"]

st.set_page_config(page_title="ZeroFile", page_icon="📄", layout="wide")


@st.cache_resource
def get_scorer() -> Scorer:
    return Scorer()


@st.cache_data
def get_report() -> dict:
    return json.loads((ARTIFACTS / "report.json").read_text(encoding="utf-8"))


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def polish(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=SECONDARY, titleColor=SECONDARY, labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor=SECONDARY, labelFontSize=12)
        .configure(background="transparent")
    )


def show(chart) -> None:
    st.altair_chart(polish(chart), width="stretch", theme=None)


def compare_lines(wide: pd.DataFrame, x: str, x_title: str, series: dict, y_title: str, kind: str = "Q", x_format: str | None = None, extra: list | None = None):
    """Two series on one axis, with a crosshair and a tooltip showing both values."""
    long = wide.melt(id_vars=[x], value_vars=list(series), var_name="column", value_name="value")
    long["Series"] = long["column"].map(series)
    axis = alt.Axis(labelAngle=0, format=x_format, tickCount=10) if x_format else alt.Axis(labelAngle=0)
    x_encoding = alt.X(f"{x}:{kind}", title=x_title, axis=axis, sort=None)
    color = alt.Color("Series:N", scale=alt.Scale(domain=list(series.values()), range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x_encoding, y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)), color=color
    )
    nearest = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    x_tip = alt.Tooltip(f"{x}:{kind}", title=x_title, format=x_format) if x_format else alt.Tooltip(f"{x}:{kind}", title=x_title)
    hover = alt.Chart(wide).mark_rule(color=MUTED, strokeWidth=1).encode(
        x=x_encoding,
        opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
        tooltip=[x_tip] + [alt.Tooltip(f"{column}:Q", title=name, format=".1%") for column, name in series.items()],
    ).add_params(nearest)
    dots = lines.mark_point(filled=True, size=70, opacity=1).transform_filter(nearest)
    return alt.layer(lines, *(extra or []), dots, hover).properties(height=300)


def score_scale(points: float, high_at: float, approve_at: float):
    bands = pd.DataFrame([
        {"start": 300, "end": high_at, "band": "High risk"},
        {"start": high_at, "end": approve_at, "band": "Human review"},
        {"start": approve_at, "end": 900, "band": "Approve"},
    ])
    bands["middle"] = (bands["start"] + bands["end"]) / 2
    x = alt.X("start:Q", scale=alt.Scale(domain=[300, 900]), axis=alt.Axis(title=None, values=[300, round(high_at), round(approve_at), 900], grid=False, domain=False, ticks=False))
    bar = alt.Chart(bands).mark_bar(height=10, cornerRadius=2).encode(
        x=x, x2="end:Q", color=alt.Color("band:N", scale=alt.Scale(domain=["High risk", "Human review", "Approve"], range=[CRITICAL, WARNING, GOOD]), legend=None), tooltip=["band:N"]
    )
    names = alt.Chart(bands).mark_text(dy=-20, fontSize=12, color=SECONDARY).encode(x=alt.X("middle:Q", scale=alt.Scale(domain=[300, 900])), text="band:N")
    marker = alt.Chart(pd.DataFrame({"points": [points]})).mark_tick(color=INK, thickness=3, size=20).encode(x=alt.X("points:Q", scale=alt.Scale(domain=[300, 900])))
    return alt.layer(bar, names, marker).properties(height=96)


def breakdown_chart(reasons: list):
    rows = pd.DataFrame(reasons)
    rows["label"] = rows["points"].map(lambda value: f"{value:+.0f}")
    y = alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=160))
    x = alt.X("points:Q", title="Points gained or lost against the average applicant", axis=alt.Axis(tickCount=6), scale=alt.Scale(padding=40))
    tip = [alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("label:N", title="Points"), alt.Tooltip("detail:N", title="Why")]
    bars = alt.Chart(rows).mark_bar(size=16, cornerRadius=3).encode(y=y, x=x, color=alt.condition(alt.datum.points >= 0, alt.value(BLUE), alt.value(ORANGE)), tooltip=tip)
    gains = alt.Chart(rows[rows["points"] >= 0]).mark_text(align="left", dx=5, color=SECONDARY).encode(y=y, x="points:Q", text="label:N")
    losses = alt.Chart(rows[rows["points"] < 0]).mark_text(align="right", dx=-5, color=SECONDARY).encode(y=y, x="points:Q", text="label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return alt.layer(bars, zero, gains, losses).properties(height=44 * len(rows) + 40)


def load_persona() -> None:
    st.session_state.update(PERSONAS[st.session_state.persona])


def applicant_tab(scorer: Scorer, report: dict) -> None:
    if "vehicle_price" not in st.session_state:
        st.session_state.update(next(iter(PERSONAS.values())))
    tier_label = lambda tier: TIER_LABELS[tier].capitalize()
    form, output = st.columns([1, 2], gap="large")
    with form:
        st.selectbox("Start from an example", PERSONAS, key="persona", on_change=load_persona)
        st.number_input("Vehicle price (₹)", min_value=10_000, max_value=5_000_000, step=1_000, key="vehicle_price")
        st.number_input("Down payment (₹)", min_value=0, max_value=5_000_000, step=500, key="down_payment")
        st.selectbox("Employment type", list(EMPLOYMENT_CODES), key="employment")
        st.selectbox("Vehicle brand's repayment record", TIER_LABELS, format_func=tier_label, key="brand")
        st.selectbox("Dealer's repayment record", TIER_LABELS, format_func=tier_label, key="dealer")
        st.selectbox("Branch's repayment record", TIER_LABELS, format_func=tier_label, key="branch")
        st.caption("A live system would look these records up from the dealer and branch. They are choices here because the dataset's IDs are anonymous.")

    applicant = Applicant(**{field: st.session_state[field] for field in ("vehicle_price", "down_payment", "employment", "brand", "dealer", "branch")})
    with output:
        try:
            result = scorer.score(applicant)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
            return
        thresholds = scorer.meta["thresholds"]
        approve_at, high_at = scorer.points_for_risk(thresholds["approve"]), scorer.points_for_risk(thresholds["high_risk"])
        band_default = {band["band"]: band["default"] for band in report["bands"]}[result.band]
        first, second, third = st.columns(3)
        first.metric("ZeroFile Score", f"{result.points:.0f}", help="300 to 900. Every 100 points halve the odds of default. This is not a CIBIL score.")
        second.metric("Estimated risk of default", pct(result.risk))
        third.metric("Loan against vehicle price", f"{applicant.ltv_pct:.0f}%")
        show(score_scale(result.points, high_at, approve_at))
        message = f"In the October back-test, no-file applicants in this band defaulted {pct(band_default)} of the time."
        if result.band == "Approve":
            st.success(f"**Approve.** {message}", icon="✅")
        elif result.band == "Human review":
            st.warning(f"**Human review.** {message}", icon="⚠️")
        else:
            st.error(f"**High risk.** A person should review this before any decline. {message}", icon="⛔")
        for note in result.warnings:
            st.info(note, icon="ℹ️")

        options = scorer.recourse(applicant)
        if result.band != "Approve":
            st.subheader("What would change this result")
            if not options:
                st.write("No change to the down payment or the vehicle alone reaches the approve band, so this needs a person.")
            for option in options:
                if option["kind"] == "down_payment":
                    st.markdown(f"**Put ₹{option['extra']:,} more down** (₹{option['down_payment']:,} in total). Estimated risk falls to {pct(option['risk'])}, which reaches the approve band.")
                else:
                    st.markdown(f"**Or choose a vehicle around ₹{option['vehicle_price']:,}** with the same down payment. Estimated risk falls to {pct(option['risk'])}.")
            if options:
                st.caption("Estimates from past loans, not a promise of approval. Identity, dealer and age are never suggested as changes.")

        st.subheader("Why this score")
        total = sum(result.factors.values())
        st.write(f"The average applicant starts at {result.base_points:.0f} points. These factors add {total:+.0f}, which gives {result.points:.0f}.")
        show(breakdown_chart(result.reasons))
        for reason in result.reasons:
            st.markdown(f"- **{reason['factor']} ({reason['points']:+.0f}):** {reason['detail']}")


def lender_tab(report: dict) -> None:
    headline = report["headline"]
    st.write(
        f"Every applicant here has no credit file, so a lender using only a bureau score approves none of them. "
        f"Move the slider to choose how many ZeroFile approves. All figures come from {headline['applicants']:,} October loans the model never trained on."
    )
    curve = pd.DataFrame(report["policy_curve"])
    share = st.slider("Approve the safest share of no-file applicants", 10, 100, 42, step=2, format="%d%%", key="share")
    row = curve.iloc[(curve["share"] - share / 100).abs().argmin()]
    first, second, third, fourth = st.columns(4)
    first.metric("People approved", f"{row['people']:,.0f}")
    second.metric("Credit unlocked", f"₹{row['crore']:,.0f} crore")
    third.metric("Their default rate", pct(row["default"]), f"{100 * (row['default'] - headline['benchmark_default']):+.1f} pts against bureau low-risk", delta_color="inverse")
    fourth.metric("LTV rule at the same size", pct(row["ltv_rule_default"]), f"{100 * (row['ltv_rule_default'] - row['default']):+.1f} pts worse than ZeroFile", delta_color="off")

    wide = curve.rename(columns={"default": "zerofile", "ltv_rule_default": "ltv_rule"})
    benchmark = alt.Chart(pd.DataFrame({"y": [headline["benchmark_default"]], "label": [f"Bureau low-risk borrowers: {pct(headline['benchmark_default'])}"], "x": [wide["share"].min()]}))
    extra = [
        benchmark.mark_rule(color=SECONDARY, strokeWidth=1).encode(y="y:Q"),
        benchmark.mark_text(align="left", dy=-8, color=SECONDARY, fontSize=12).encode(x="x:Q", y="y:Q", text="label:N"),
        alt.Chart(pd.DataFrame([row])).mark_point(filled=True, size=140, color=BLUE, stroke="#fcfcfb", strokeWidth=2, opacity=1).encode(x="share:Q", y="default:Q"),
    ]
    show(compare_lines(wide, "share", "Share of no-file applicants approved", {"zerofile": "ZeroFile", "ltv_rule": "Simple LTV rule"}, "Default rate among those approved", x_format=".0%", extra=extra))
    st.caption(
        f"The slider shows the whole range after the fact. The headline result is stricter: its cutoff was fixed on September data, before October was scored, and approves {pct(headline['share'])} of applicants at a {pct(headline['default'])} default rate. "
        "Every row is a loan the lender actually funded, so this is a back-test."
    )
    with st.expander("See these numbers as a table"):
        table = pd.DataFrame({
            "Approved": curve["share"].map(lambda v: pct(v, 0)),
            "People": curve["people"],
            "Credit (₹ crore)": curve["crore"].round(1),
            "ZeroFile default rate": curve["default"].map(pct),
            "LTV rule default rate": curve["ltv_rule_default"].map(pct),
        })
        st.dataframe(table, hide_index=True, width="stretch")


def evidence_tab(report: dict) -> None:
    auc, periods = report["auc"], report["periods"]
    no_bureau, gain = auc["no_bureau"], auc["rated_gain"]
    classification = report["headline"]["classification"]
    first, second, third, fourth = st.columns(4)
    first.metric("AUC, no-file applicants", f"{no_bureau['value']:.3f}", help=f"95% bootstrap interval {no_bureau['low']:.3f} to {no_bureau['high']:.3f}. A bureau score cannot rank these people at all, because they have none.")
    second.metric("KS statistic", f"{report['ks_no_bureau']:.3f}")
    third.metric("Brier score", f"{report['brier_no_bureau']:.3f}", f"Base rate alone: {report['brier_base_rate_no_bureau']:.3f}", delta_color="off")
    fourth.metric("Scoring time", f"{report['latency_ms_p50']:.1f} ms", help="Median for one applicant, including the points breakdown.")
    st.caption(
        f"Trained on {periods['train']['rows']:,} loans ({periods['train']['start']} to {periods['train']['end']}), tuned on {periods['validation']['rows']:,} ({periods['validation']['start']} to {periods['validation']['end']}), "
        f"and tested on {periods['test']['rows']:,} from {periods['test']['start']} to {periods['test']['end']}. The split is by date because default rates drift by month."
    )

    st.subheader("It out-ranks the bureau score, even where a score exists")
    st.write(
        f"On borrowers with a real bureau grade, ZeroFile reaches AUC {auc['rated_zerofile']['value']:.3f} against {auc['rated_bureau_score']['value']:.3f} for the bureau score itself. "
        f"The gain is {gain['value']:.3f} (95% CI {gain['low']:.3f} to {gain['high']:.3f}). Full bureau data is still valuable where it exists, as the last row below shows, so ZeroFile is a way in for people without a file and not a replacement for the bureau."
    )
    variants = pd.DataFrame(report["variants"])
    st.dataframe(
        pd.DataFrame({
            "Model": variants["name"],
            "AUC, no-file applicants": variants["auc_no_bureau"].map("{:.3f}".format),
            "AUC, bureau-rated borrowers": variants["auc_rated"].map("{:.3f}".format),
            "Lowest regional approval, as a share of the highest": variants["worst_region_ratio"].map(lambda v: pct(v, 0)),
        }),
        hide_index=True,
        width="stretch",
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Risk ranking and calibration")
        rows = pd.DataFrame(report["calibration"])
        rows["decile"] = range(1, len(rows) + 1)
        show(compare_lines(rows, "decile", "Risk decile, safest to riskiest", {"actual": "Actual default rate", "predicted": "Predicted"}, "Default rate", kind="O"))
        st.caption(f"The riskiest tenth defaults {rows['actual'].iloc[-1] / rows['actual'].iloc[0]:.1f} times as often as the safest. Expected calibration error: {pct(report['ece_no_bureau'])}.")
    with right:
        st.subheader("Predictions ran low as October went on")
        weeks = pd.DataFrame(report["weekly_drift"])
        weeks["week"] = weeks["week"].map(WEEK_STARTS)
        show(compare_lines(weeks, "week", "Week starting", {"actual": "Actual default rate", "predicted": "Predicted"}, "Default rate", kind="N"))
        st.caption("Defaults jumped at the end of October, and the model did not see that coming. Ranking held up better than absolute risk, so a live system would need regular recalibration.")

    st.subheader("What moves the score most")
    importance = pd.DataFrame(report["importance"])
    bars = alt.Chart(importance).mark_bar(size=16, cornerRadiusEnd=4, color=BLUE).encode(
        y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False)),
        x=alt.X("points:Q", title="Average effect on the score, in points", axis=alt.Axis(tickCount=6)),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("points:Q", title="Average points", format=".1f")],
    )
    show(bars.properties(height=44 * len(importance) + 40))
    st.caption(
        f"Treating 'not approved' as a prediction of default gives recall {pct(classification['recall'], 0)}, precision {pct(classification['precision'], 0)}, F1 {classification['f1']:.2f} and accuracy {pct(classification['accuracy'], 0)}. "
        "The lender picks this trade-off with the slider, so ranking and default rate matter more here than a single accuracy figure."
    )


def fairness_tab(report: dict) -> None:
    fairness, floor, headline = report["fairness"], report["regional_floor"], report["headline"]
    regions = fairness["Region"]
    failing = [row for row in regions if not row["passes_four_fifths"]]
    st.write(
        "Age, region and ID documents are **not model inputs**. This audit checks the result anyway, because other inputs can stand in for them. "
        "The test is the four-fifths rule: a group is flagged if its approval rate is under 80% of the best group's."
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

    st.subheader("A measured fix: a regional approval floor")
    st.write(f"Require every region to approve at least {pct(floor['floor'])} of its no-file applicants (80% of the overall rate), safest applicants first.")
    st.dataframe(
        pd.DataFrame({
            "Policy": ["Model cutoff alone", "With the regional floor"],
            "People approved": [f"{headline['people']:,}", f"{floor['people']:,}"],
            "Credit (₹ crore)": [f"{headline['crore']:.0f}", f"{floor['crore']:.0f}"],
            "Default rate": [pct(headline["default"]), pct(floor["default"])],
            "Regions under the floor": [floor["regions_below_before"], floor["regions_below_after"]],
            "Lowest regional approval, as a share of the highest": [pct(min(row["ratio"] for row in regions), 0), pct(floor["worst_ratio"], 0)],
        }),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"The floor brings in {floor['people_added']:,} more people and stays below the {pct(headline['benchmark_default'])} bureau low-risk benchmark. It narrows the regional gap but does not close it, and it was computed on October after the fact.")

    st.subheader("Full audit")
    name = st.radio("Group by", list(fairness), horizontal=True, key="audit_dimension")
    rows = pd.DataFrame(fairness[name])
    if name == "Age band":
        rows = rows.set_index("group").loc[AGE_ORDER].reset_index()
    elif name == "Region":
        rows = rows.sort_values("approval", ascending=False)
        rows["group"] = "Region " + rows["group"]
    rows["status"] = rows["passes_four_fifths"].map({True: "Passes", False: "Below four-fifths"})
    line = rows["approval"].max() * 0.8
    y = alt.Y("group:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False))
    bars = alt.Chart(rows).mark_bar(size=14, cornerRadiusEnd=4).encode(
        y=y,
        x=alt.X("approval:Q", title="Approval rate", axis=alt.Axis(format=".0%", tickCount=8)),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Passes", "Below four-fifths"], range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("approval:Q", title="Approved", format=".1%"), alt.Tooltip("default:Q", title="Actual default rate", format=".1%"), alt.Tooltip("status:N", title="Four-fifths rule")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(color=SECONDARY, strokeWidth=1).encode(x="x:Q")
    show(alt.layer(bars, rule).properties(height=32 * len(rows) + 70))
    st.caption(f"The vertical line marks four-fifths of the best group's approval rate ({pct(line)}).")
    st.dataframe(
        pd.DataFrame({
            "Group": rows["group"],
            "Applicants": rows["n"],
            "Approved": rows["approval"].map(pct),
            "Actual default rate": rows["default"].map(pct),
            "Default rate if approved": rows["default_if_approved"].map(pct),
            "Approved among those who repaid": rows["approval_if_repaid"].map(pct),
            "Four-fifths rule": rows["status"],
        }),
        hide_index=True,
        width="stretch",
    )


def limits_tab(report: dict) -> None:
    st.subheader("What this is")
    st.write("A decision aid and research prototype. It should send borderline cases to a person, and it should never decline someone on its own.")
    st.subheader("Known limits")
    st.markdown(
        """
- **It is a back-test.** Every row is a loan the lender funded. Nobody knows how rejected applicants would have behaved, so results may not carry over to them.
- **Thin-file, not fully unbanked.** These are 2018 vehicle-loan borrowers with no bureau record. Most probably have bank accounts. Reaching the fully unbanked would take new data such as cash-flow records.
- **Modest signal.** First-instalment default is noisy, and AUC stays near 0.63. The model separates groups well, but it cannot predict any single person.
- **Risk estimates ran low.** October's defaults rose beyond what the model expected. The bands show what actually happened in October for that reason.
- **Region fails the fairness audit.** See the Fairness tab. Dealer and branch records partly stand in for location.
- **Not a fully blind test.** October was inspected while choosing features. Those choices only removed inputs, which should not inflate results, but a truly untouched test does not exist here.
- **Not causal.** A bigger down payment lowers the model's estimate. It does not prove that person will repay.
        """
    )
    st.subheader("Privacy and data")
    st.markdown(
        """
- Date of birth, pincode, customer ID and employee code are dropped as the data loads.
- The deployed app holds a model and aggregate results only. It stores nothing you type, and the examples are fictional.
- Data: [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) on Kaggle. Its licence is unclear, so raw data is not redistributed.
        """
    )


if not (ARTIFACTS / "model.txt").exists() or not (ARTIFACTS / "report.json").exists():
    st.error("The model files are missing. Run `python -m zerofile.data`, then `python -m zerofile.train`.", icon="🚫")
    st.stop()

scorer, report = get_scorer(), get_report()
headline, population = report["headline"], report["population"]

st.title("ZeroFile")
st.markdown("**A credit score for people with no credit file.** A bureau score cannot rank someone it has never seen. ZeroFile scores them from the loan application, explains every result, and shows what would change it.")
first, second, third, fourth = st.columns(4)
first.metric("Borrowers with no credit file", pct(population["no_bureau_share"]))
first.caption(f"{population['no_bureau']:,} of {population['rows']:,} loans in a public Indian vehicle-loan dataset")
second.metric("Approved in the October back-test", f"{headline['people']:,}")
second.caption(f"{pct(headline['share'])} of no-file applicants. A score-only lender approves none.")
third.metric("Credit unlocked in one month", f"₹{headline['crore']:.0f} crore")
third.caption("Cutoff fixed on September data, before October was scored")
fourth.metric("Their default rate", pct(headline["default"]))
fourth.caption(f"Bureau low-risk borrowers: {pct(headline['benchmark_default'])}. All no-file applicants: {pct(headline['no_bureau_default'])}.")

applicant, lender, evidence, fairness, limits = st.tabs(["Score an applicant", "Lender impact", "Evidence", "Fairness audit", "Limits and data"])
with applicant:
    applicant_tab(scorer, report)
with lender:
    lender_tab(report)
with evidence:
    evidence_tab(report)
with fairness:
    fairness_tab(report)
with limits:
    limits_tab(report)

st.divider()
st.caption(DISCLAIMER)
