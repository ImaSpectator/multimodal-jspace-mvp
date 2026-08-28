from pathlib import Path

from backend.jspace import ai_provider
from backend.jspace.ai_provider import _enforce_simulation_reply
from backend.jspace.scenario_generator import generate_manual_context, generate_scenario
from backend.jspace.schemas import ScenarioControls
from backend.jspace.simulator import apply_manual_customer_message, new_manual_state

ROOT = Path(__file__).parents[1]


def test_v142_scenario_customer_text_has_clean_sentence_capitalization():
    for seed in range(1, 15):
        scenario = generate_scenario(ScenarioControls(domain="hotel_hospitality", seed=seed))
        for step in scenario.steps:
            text = step.customer_turn.text
            assert " because i " not in text.lower()
            assert ": i " not in text.lower()
            assert ". can you" not in text.lower()
            assert "Good, we're getting somewhere" not in text


def test_v142_manual_resolution_request_completes_simulated_work_without_wait_loop():
    profile, backend = generate_manual_context("account_access", seed=9)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, profile["_manual_case"]["opening"])
    apply_manual_customer_message(state, profile["_manual_case"]["impact"])
    apply_manual_customer_message(state, "Please fix this from your side.")
    status = next(c for c in state.concepts if c.name == "authoritative_status")
    assert status.value == "resolved"
    assert state.session_phase == "resolved"

    bad_model_reply = "I'm pulling the backend now and I'll update you when I confirm it."
    safe = _enforce_simulation_reply(state, bad_model_reply, language="English")
    assert "resolved" in safe.lower()
    assert "i'm pulling" not in safe.lower()
    assert "i’ll update" not in safe.lower()


def test_v142_manual_suggestions_do_not_contain_simulation_meta_language():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert "this is turn 7" not in source.lower()
    assert "this is turn 13" not in source.lower()
    assert "carry out the next concrete action" not in source.lower()
    assert "what have you verified so far" not in source.lower()
    assert "manual version of Scenario Lab" in source
    assert "Okay, that makes sense. Please go ahead and fix it." in source


def test_v142_conflict_severity_labels_are_bilingual_and_include_priority():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert '"high": ("HIGH PRIORITY SIGNAL CONFLICT", "高优先级信号冲突")' in source
    assert '"medium": ("MEDIUM PRIORITY SIGNAL CONFLICT", "中优先级信号冲突")' in source
    assert '"low": ("LOW PRIORITY SIGNAL CONFLICT", "低优先级信号冲突")' in source
    assert "display_conflict_severity(conflict.severity)" in source


def test_v142_pdf_is_not_a_chat_bubble_replica():
    source = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert "Conversation" in source
    assert "Conversation Analysis" in source
    assert "ordinary flowing text" in source
    assert "chat bubbles, cards, or tables" in source
    assert "Table(" not in source
