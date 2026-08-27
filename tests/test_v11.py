from pathlib import Path

from backend.jspace.scenario_generator import generate_manual_context
from backend.jspace.simulator import new_manual_state, update_customer_relationship, apply_manual_customer_message


def _source():
    return (Path(__file__).parents[1] / "frontend" / "app.py").read_text()


def test_v11_borderless_toolbar_and_link_icon():
    source = _source()
    assert 'APP_VERSION = "1.1.0-behavior-polish"' in source
    assert ':material/link:' in source
    assert 'background:transparent!important; box-shadow:none!important;' in source
    assert 'border:0!important; border-radius:999px!important;' in source
    assert 'align-items:center!important; justify-content:center!important;' in source


def test_v11_suggested_moves_are_not_question_only():
    source = _source()
    assert 'Suggest the customer\'s next *move*, not just another question.' in source
    assert 'Please go ahead with the concrete fix' in source
    assert 'please skip the repeats and move to the next system-side check' in source
    assert 'That works on my side now. I don\'t have any other concerns' in source


def test_v11_chinese_concepts_have_name_and_value_localization():
    source = _source()
    assert 'CONCEPT_NAME_ZH' in source
    assert '"root_cause": "根本原因"' in source
    assert '"authoritative_status": "权威系统状态"' in source
    assert 'CONCEPT_VALUE_ZH' in source
    assert '"merchant category restriction": "商户类别限制"' in source
    assert 'display_concept_name(c.name)' in source
    assert 'display_concept_value(c.name, c.value)' in source


def test_v11_patience_starts_full_and_relationship_updates():
    profile, backend = generate_manual_context("payment", seed=42)
    assert profile["patience"] == 100
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    original_trust = profile["trust"]
    update_customer_relationship(profile, state, "I'm checking the latest state.", "Local fallback after bounded retry (TimeoutError)")
    assert profile["patience"] < 100
    assert profile["trust"] <= original_trust


def test_v11_progress_does_not_consume_patience_and_can_raise_trust():
    profile, backend = generate_manual_context("account_access", seed=7)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    original_trust = profile["trust"]
    update_customer_relationship(profile, state, "I've identified the root cause and the next step is to unlock the account.", "DeepSeek · model")
    assert profile["patience"] == 100
    assert profile["trust"] >= original_trust


def test_v11_manual_authorized_fix_can_progress_to_resolution():
    profile, backend = generate_manual_context("payment", seed=11)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, "Please check the blocker first.")
    apply_manual_customer_message(state, "That makes sense. Please go ahead with that fix.")
    authoritative = next(c for c in state.concepts if c.name == "authoritative_status")
    assert authoritative.value == "resolved"
    assert state.session_phase == "resolved"


def test_v11_settings_are_stored_outside_dialog_widget_keys():
    source = _source()
    assert 'st.session_state.setdefault("app_settings", dict(SETTINGS_DEFAULTS))' in source
    assert 'def _persist_setting(widget_key: str, name: str)' in source
    assert 'def _setting(name: str)' in source
    assert 'on_change=_persist_setting' in source
    assert '_setting("researcher_view")' in source
    assert '_setting("auto_scroll")' in source


def test_v11_media_evidence_requests_selected_language():
    source = (Path(__file__).parents[1] / "backend" / "jspace" / "ai_provider.py").read_text()
    assert 'language: str = "English"' in source
    assert 'Write every summary in {output_language}' in source
    assert 'Write all three values in {output_language}' in source
    assert 'temperature": 0.55' in source
