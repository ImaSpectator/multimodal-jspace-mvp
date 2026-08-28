from pathlib import Path

from backend.jspace.ai_provider import build_support_prompt
from backend.jspace.scenario_generator import generate_manual_context
from backend.jspace.simulator import apply_manual_customer_message, new_manual_state

ROOT = Path(__file__).parents[1]


def test_v141_support_prompt_reports_check_result_instead_of_waiting():
    profile, backend = generate_manual_context("hotel_hospitality", seed=7)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, "My upgrade is missing from the confirmation.")
    prompt = build_support_prompt(state, profile, "hotel_hospitality")
    assert 'Never say "I\'m checking"' in prompt
    assert "completed result of any support-side lookup" in prompt
    assert "If authoritative_status is unresolved" in prompt


def test_v141_manual_can_resolve_on_natural_fourth_customer_turn():
    profile, backend = generate_manual_context("hotel_hospitality", seed=7)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, profile["_manual_case"]["opening"])
    apply_manual_customer_message(state, profile["_manual_case"]["impact"])
    apply_manual_customer_message(state, profile["_manual_case"]["followup"])
    assert not any(c.name == "root_cause" for c in state.concepts)
    apply_manual_customer_message(state, "What did you find, and can you fix the actual cause from your side?")
    assert any(c.name == "root_cause" for c in state.concepts)
    apply_manual_customer_message(state, "Okay, that makes sense. Please go ahead and fix it.")
    assert next(c for c in state.concepts if c.name == "authoritative_status").value == "resolved"
    assert state.session_phase == "resolved"


def test_v141_manual_suggestions_avoid_meta_diagnostic_language():
    src = (ROOT / "frontend" / "app.py").read_text()
    banned = [
        "What have you verified so far, and which single check comes next?",
        "Please continue that check and tell me what changes once you verify the blocker.",
        "What is the most useful thing you can verify next without making me repeat earlier information?",
    ]
    for text in banned:
        assert text not in src
    assert "Please go ahead and fix it." in src
    assert "Acknowledge and close" not in src


def test_v141_settings_gear_moved_left_two_pixels_from_v140():
    src = (ROOT / "frontend" / "app.py").read_text()
    block = src.split('.st-key-top_settings [data-testid="stIconMaterial"]', 1)[1].split('}', 1)[0]
    assert "translateX(4px)" in block
