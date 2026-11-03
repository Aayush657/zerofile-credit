"""Runs the real Streamlit script headlessly, so a broken page fails the suite."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PERSONAS = [
    ("Self-employed delivery rider, a close call", "Human review"),
    ("Salaried clinic assistant, larger down payment", "Approve"),
    ("Self-employed tailor, small down payment", "High risk"),
]
WORKERS = [
    ("Rural farm worker, no UPI", "Human review"),
    ("Gig delivery rider, steady habits", "Approve"),
    ("Seasonal worker, missed bills", "High risk"),
]


def start() -> AppTest:
    return AppTest.from_file(APP, default_timeout=90).run()


def all_text(app: AppTest) -> str:
    parts = [element.value for group in (app.markdown, app.success, app.warning, app.error, app.caption, app.info) for element in group]
    return "\n".join(parts)


def band_messages(app: AppTest) -> str:
    return "\n".join(element.value for group in (app.success, app.warning, app.error) for element in group)


def scores(app: AppTest) -> list:
    return [metric.value for metric in app.metric if metric.label == "ZeroFile Score"]


def test_app_loads_with_every_tab():
    app = start()
    assert not app.exception
    assert app.title[0].value == "ZeroFile"
    assert [tab.label for tab in app.tabs] == ["Score a worker", "Score a vehicle loan", "Lender impact", "Evidence", "Fairness audit", "Limits and data"]


@pytest.mark.parametrize("persona,band", PERSONAS)
def test_each_vehicle_example_reaches_its_band(persona, band):
    app = start()
    app.selectbox(key="persona").select(persona).run()
    assert not app.exception
    assert any(line.startswith(f"**{band}.**") and "October back-test" in line for line in band_messages(app).split("\n"))


@pytest.mark.parametrize("worker,band", WORKERS)
def test_each_worker_example_reaches_its_band(worker, band):
    app = start()
    app.selectbox(key="worker_example").select(worker).run()
    assert not app.exception
    assert any(line.startswith(f"**{band}.**") and "synthetic test set" in line for line in band_messages(app).split("\n"))


def test_close_call_shows_recourse_and_approved_case_does_not():
    app = start()
    assert "more down" in all_text(app)
    app.selectbox(key="persona").select(PERSONAS[1][0]).run()
    assert "more down" not in all_text(app)


def test_no_upi_worker_is_flagged_and_uses_a_person():
    text = all_text(start())
    assert "no UPI history" in text
    assert "Pay 60% of utility bills on time" in text


def test_worker_toggles_survive_switching_off_and_on():
    app = start()
    for value in (True, False, True):
        app.toggle(key="w_uses_upi").set_value(value).run()
        assert not app.exception
    assert app.slider(key="w_upi_payments").value == 30
    for value in (True, False, True):
        app.toggle(key="w_has_rent").set_value(value).run()
        assert not app.exception
    assert app.slider(key="w_rent").value == 50


def test_changing_a_worker_changes_the_score():
    app = start()
    before = scores(app)[0]
    app.slider(key="w_bills").set_value(95).run()
    assert not app.exception and scores(app)[0] != before


def test_loan_range_is_shown_for_both_models():
    app = start()
    assert "Up to ₹124,000 goes to human review" in all_text(app)
    assert "Up to ₹46,500" in all_text(app)


def test_invalid_down_payment_shows_a_message_not_a_crash():
    app = start()
    app.number_input(key="down_payment").set_value(500_000).run()
    assert not app.exception
    assert "less than the vehicle price" in all_text(app)


def test_out_of_range_vehicle_warns():
    app = start()
    app.number_input(key="vehicle_price").set_value(4_000_000).run()
    assert not app.exception
    assert any("outside" in element.value for element in app.info)


def test_slider_updates_the_lender_view():
    app = start()
    before = next(metric.value for metric in app.metric if metric.label == "People approved")
    app.slider(key="share").set_value(70).run()
    after = next(metric.value for metric in app.metric if metric.label == "People approved")
    assert not app.exception and before != after


def test_lender_economics_respond_to_the_assumptions():
    app = start()
    best = lambda: next(metric for metric in app.metric if metric.label == "Best approval level for these numbers").value
    before = best()
    app.number_input(key="margin").set_value(30).run()
    assert not app.exception and best() != before


def test_lender_view_labels_the_simulation():
    text = all_text(start())
    assert "This is a simulation, not a result" in text
    assert "Closest group to gig workers" in text


def test_evidence_warns_that_worker_accuracy_is_not_real_world():
    assert "They do not show real-world accuracy" in all_text(start())


@pytest.mark.parametrize("dimension", ["Age band", "Employment", "Region"])
def test_real_fairness_views_render(dimension):
    app = start()
    app.radio(key="audit_dimension").set_value(dimension).run()
    assert not app.exception


@pytest.mark.parametrize("dimension", ["Worker type", "Employment", "State", "UPI access"])
@pytest.mark.parametrize("outcome", ["Not marked high risk", "Approved outright"])
def test_worker_fairness_views_render(dimension, outcome):
    app = start()
    app.radio(key="fairness_model").set_value("Worker model").run()
    app.radio(key="worker_audit_dimension").set_value(dimension).run()
    app.radio(key="worker_audit_outcome").set_value(outcome).run()
    assert not app.exception
    assert "Gender is not in this dataset" in all_text(app)


def test_disclaimer_and_limits_are_always_visible():
    text = all_text(start())
    assert "Not a lending decision or financial advice" in text
    assert "Synthetic labels" in text
