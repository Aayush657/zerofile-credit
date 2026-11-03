"""Submission rules that must never regress: required README content, no raw data, and numbers that match the report."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORT = json.loads((ROOT / "artifacts" / "report.json").read_text(encoding="utf-8"))


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
    text = (ROOT / "artifacts" / "report.json").read_text(encoding="utf-8") + (ROOT / "artifacts" / "meta.json").read_text(encoding="utf-8")
    for field in ("UniqueID", "Date.of.Birth", "pincode", "Employee_code", "supplier_id"):
        assert field not in text


@pytest.mark.parametrize("required", [
    "HI-04",
    "Human Impact",
    "Aayush Sharma",
    "## Problem",
    "## Solution",
    "## Tech stack",
    "## Setup and running locally",
    "## Screenshots",
    "## Results",
    "## AI tools used",
    "Claude",
    "```mermaid",
    "## Future scope",
])
def test_readme_has_required_content(required):
    assert required in README


def test_readme_screenshots_exist():
    for name in ("applicant", "lender", "evidence", "fairness"):
        assert f"docs/screenshots/{name}.png" in README
        assert (ROOT / "docs" / "screenshots" / f"{name}.png").exists()


def test_readme_numbers_match_the_report():
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
        f"{REPORT['population']['no_bureau']:,}",
        f"{REPORT['regional_floor']['people_added']:,}",
    ]
    for row in REPORT["alt_scenarios"]:
        expected += [f"{row['scenario']} (simulated)", f"{row['people']:,}", f"₹{row['crore']:.0f} crore", f"{row['auc_no_bureau']:.3f}", f"{100 * row['share']:.1f}%", f"{100 * row['default']:.1f}%"]
    gig = next(row for row in REPORT["fairness"]["Employment"] if row["group"] == "Self employed")
    expected += [f"{gig['n']:,} applicants", f"{100 * gig['approval']:.1f}% approved", f"{100 * gig['default_if_approved']:.1f}% default"]
    for number in expected:
        assert number in README, number


def test_readme_has_the_live_link_and_no_placeholders():
    assert "https://zerofile.streamlit.app" in README
    assert "LIVE_DEMO_URL" not in README and "DEMO_VIDEO_URL" not in README


def test_the_official_brief_is_addressed_and_the_simulation_is_labelled():
    assert "## How this answers the official brief" in README
    assert "**This is a simulation, not a result.**" in README
    assert "Gender" in README and "rural" in README


def test_no_commit_trailer_template_or_docs_credit_claude_as_a_contributor():
    for name in ("docs/SUBMISSION.md", "docs/DEPLOY.md", "docs/DEMO_SCRIPT.md"):
        assert "Co-Authored-By" not in (ROOT / name).read_text(encoding="utf-8")
