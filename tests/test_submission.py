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
    for number in expected:
        assert number in README, number
