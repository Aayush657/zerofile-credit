"""Submission rules that must never regress: required README content, no raw data, and numbers that match the reports."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORT = json.loads((ROOT / "artifacts" / "report.json").read_text(encoding="utf-8"))
WORKER = json.loads((ROOT / "artifacts" / "worker_report.json").read_text(encoding="utf-8"))
TUNING = json.loads((ROOT / "artifacts" / "tuning.json").read_text(encoding="utf-8"))


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("Not a git checkout")
    return result.stdout.splitlines()


def test_no_raw_data_or_secrets_are_tracked():
    for name in tracked_files():
        assert not name.endswith((".csv", ".zip", ".parquet")), name
        assert not name.startswith("data/"), name
        assert Path(name).name != ".env", name


def test_artifacts_hold_no_borrower_level_fields():
    text = "".join(
        (ROOT / "artifacts" / name).read_text(encoding="utf-8")
        for name in ("report.json", "meta.json", "worker_report.json", "worker_meta.json", "tuning.json")
    )
    for field in ("UniqueID", "Date.of.Birth", "pincode", "Employee_code", "supplier_id", "borrower_id", "B00001"):
        assert field not in text


def test_commit_history_credits_only_the_author():
    result = subprocess.run(["git", "log", "--format=%B"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("Not a git checkout")
    assert "co-authored-by" not in result.stdout.lower()


@pytest.mark.parametrize(
    "required",
    [
        "HI-04",
        "Human Impact",
        "Aayush Sharma",
        "ZeroFile (solo entry)",
        "The problem.",
        "What ZeroFile does.",
        "## Tech stack",
        "## Setup and running locally",
        "## Screenshots",
        "## Results: vehicle-loan model",
        "## Results: worker model",
        "## AI tools used",
        "Claude",
        "```mermaid",
        "## Future scope",
        "docs/demo.gif",
    ],
)
def test_readme_has_required_content(required):
    assert required in README


def test_readme_screenshots_and_gif_exist():
    for name in ("worker", "applicant", "lender", "evidence", "fairness", "fairness_worker"):
        assert f"docs/screenshots/{name}.png" in README
        assert (ROOT / "docs" / "screenshots" / f"{name}.png").exists()
    assert (ROOT / "docs" / "demo.gif").exists()


def test_readme_numbers_match_the_real_data_report():
    headline, auc = REPORT["headline"], REPORT["auc"]
    expected = [
        f"{headline['people']:,}",
        f"{headline['applicants']:,}",
        f"{100 * headline['share']:.1f}%",
        f"₹{headline['crore']:.0f} crore",
        f"{100 * headline['default']:.1f}%",
        f"{100 * headline['benchmark_default']:.1f}%",
        f"{100 * headline['ltv_rule_default']:.1f}%",
        f"{auc['no_bureau']['value']:.3f}",
        f"{auc['rated_zerofile']['value']:.3f}",
        f"{auc['rated_bureau_score']['value']:.3f}",
        f"{REPORT['logistic_baseline']['auc_no_bureau']:.3f}",
        f"{REPORT['population']['no_bureau']:,}",
        f"{REPORT['regional_floor']['people_added']:,}",
        f"{TUNING['configurations']} LightGBM settings",
        f"between {TUNING['auc_test_low']:.3f} and {TUNING['auc_test_high']:.3f}",
    ]
    for row in REPORT["alt_scenarios"]:
        expected += [
            f"{row['scenario']} (simulated)",
            f"{row['people']:,}",
            f"₹{row['crore']:.0f} crore",
            f"{row['auc_no_bureau']:.3f}",
            f"{100 * row['share']:.1f}%",
            f"{100 * row['default']:.1f}%",
        ]
    gig = next(row for row in REPORT["fairness"]["Employment"] if row["group"] == "Self employed")
    expected += [
        f"{gig['n']:,} applicants",
        f"{100 * gig['approval']:.1f}% approved",
        f"{100 * gig['default_if_approved']:.1f}% default",
    ]
    for number in expected:
        assert number in README, number


def test_readme_numbers_match_the_worker_report():
    def find(rows, group):
        return next(row for row in rows if row["group"] == group)

    auc = WORKER["auc"]
    expected = [
        f"{auc['value']:.3f} ({auc['low']:.3f} to {auc['high']:.3f})",
        f"{min(WORKER['auc_by_split']):.3f} to {max(WORKER['auc_by_split']):.3f}",
        f"{WORKER['auc_logistic']:.3f}",
        f"{WORKER['brier']:.3f}",
        f"{100 * WORKER['ece']:.1f}%",
    ]
    for band in WORKER["bands"]:
        expected += [f"{100 * band['share']:.1f}%", f"{100 * band['default']:.1f}%"]
    for variant in WORKER["variants"]:
        expected.append(f"{variant['auc']:.3f}")
    rural, rural_reach = (
        find(WORKER["fairness"]["Worker type"], "rural"),
        find(WORKER["fairness_not_high_risk"]["Worker type"], "rural"),
    )
    no_upi, no_upi_reach = (
        find(WORKER["fairness"]["UPI access"], "No UPI history"),
        find(WORKER["fairness_not_high_risk"]["UPI access"], "No UPI history"),
    )
    expected += [
        f"{100 * rural['approval']:.0f}% (fails)",
        f"{100 * rural_reach['approval']:.0f}% (passes)",
        f"{100 * no_upi['approval']:.0f}% (fails)",
        f"{100 * no_upi_reach['approval']:.0f}% (fails)",
    ]
    expected.append(f"{100 * WORKER['smaller_loan']['rescued_default_on_loan_asked']:.0f}% of those people defaulted")
    for number in expected:
        assert number in README, number


def test_readme_has_the_live_link_and_no_placeholders():
    assert "https://zerofile.streamlit.app" in README
    assert "LIVE_DEMO_URL" not in README and "DEMO_VIDEO_URL" not in README


def test_the_official_brief_is_addressed_and_synthetic_results_are_labelled():
    assert "## How this answers the official brief" in README
    assert "**This is a simulation, not a result.**" in README
    assert "**These numbers show the pipeline works. They do not show real-world accuracy.**" in README
    assert "Gender" in README and "rural" in README


def test_docs_do_not_credit_claude_as_a_contributor():
    for name in ("docs/SUBMISSION.md", "docs/DEPLOY.md", "docs/DEMO_SCRIPT.md"):
        assert "Co-Authored-By" not in (ROOT / name).read_text(encoding="utf-8")
