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


def start() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def all_text(app: AppTest) -> str:
    parts = [element.value for group in (app.markdown, app.success, app.warning, app.error, app.caption) for element in group]
    return "\n".join(parts)


def test_app_loads_with_every_tab():
    app = start()
    assert not app.exception
    assert app.title[0].value == "ZeroFile"
    assert [tab.label for tab in app.tabs] == ["Score an applicant", "Lender impact", "Evidence", "Fairness audit", "Limits and data"]


@pytest.mark.parametrize("persona,band", PERSONAS)
def test_each_example_reaches_its_band(persona, band):
    app = start()
    app.selectbox(key="persona").select(persona).run()
    assert not app.exception
    assert f"**{band}" in all_text(app)


def test_close_call_shows_recourse_and_approved_case_does_not():
    app = start()
    assert "more down" in all_text(app)
    app.selectbox(key="persona").select(PERSONAS[1][0]).run()
    assert "more down" not in all_text(app)


def test_loan_range_is_shown():
    assert "stays in the approve band" in all_text(start())


def test_simulated_toggle_is_labelled_and_changes_the_result():
    app = start()
    before = next(metric.value for metric in app.metric if metric.label == "ZeroFile Score")
    assert "back-test, no-file applicants in this band" in all_text(app)
    app.toggle(key="use_alt").set_value(True).run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "ZeroFile Score") != before
    assert "simulated signals" in all_text(app)
    assert "back-test, no-file applicants in this band" not in all_text(app)
    assert app.slider(key="bills_on_time").value == 90


def test_simulated_toggle_survives_switching_off_and_on():
    app = start()
    app.toggle(key="use_alt").set_value(True).run()
    app.toggle(key="use_alt").set_value(False).run()
    assert not app.exception
    app.toggle(key="use_alt").set_value(True).run()
    assert not app.exception and app.slider(key="bills_on_time").value == 90


def test_lender_view_labels_the_simulation():
    text = all_text(start())
    assert "This is a simulation, not a result" in text
    assert "Closest group to gig workers" in text


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


@pytest.mark.parametrize("dimension", ["Age band", "Employment", "Region"])
def test_fairness_views_render(dimension):
    app = start()
    app.radio(key="audit_dimension").set_value(dimension).run()
    assert not app.exception


def test_disclaimer_is_always_visible():
    assert "Not a lending decision or financial advice" in all_text(start())
