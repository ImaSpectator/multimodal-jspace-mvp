from pathlib import Path

from backend.jspace.ai_provider import build_support_prompt
from backend.jspace.conversation_export import build_conversation_pdf
from backend.jspace.scenario_generator import generate_manual_context
from backend.jspace.simulator import apply_manual_customer_message, new_manual_state

ROOT = Path(__file__).parents[1]


def test_v14_manual_diagnosis_is_progressive_not_preloaded():
    profile, backend = generate_manual_context("payment", seed=11)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    assert not any(c.name == "root_cause" for c in state.concepts)
    apply_manual_customer_message(state, "My payment is failing.")
    apply_manual_customer_message(state, "I need this today.")
    assert not any(c.name == "root_cause" for c in state.concepts)
    apply_manual_customer_message(state, "I've retried it already.")
    assert any(c.name == "root_cause" for c in state.concepts)


def test_v14_manual_cannot_collapse_to_three_turn_resolution():
    profile, backend = generate_manual_context("account_access", seed=3)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, "I can't log in.")
    apply_manual_customer_message(state, "Please go ahead and fix it.")
    apply_manual_customer_message(state, "I already reset the password. Please do it.")
    assert next(c for c in state.concepts if c.name == "authoritative_status").value == "unresolved"
    apply_manual_customer_message(state, "What will the change do?")
    apply_manual_customer_message(state, "Okay, please go ahead with the fix.")
    assert next(c for c in state.concepts if c.name == "authoritative_status").value == "resolved"


def test_v14_manual_agent_prompt_has_real_conversation_pacing():
    profile, backend = generate_manual_context("delivery", seed=4)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    apply_manual_customer_message(state, "My package isn't here.")
    prompt = build_support_prompt(state, profile, "delivery")
    assert "early manual-practice discovery turn" in prompt
    assert "Do not rush to a fix or closure" in prompt


def test_v14_pdf_uses_independent_full_width_cards():
    src = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert "archive-friendly conversation report" in src
    assert "PageBreak()" in src
    assert "Customer and support messages are preserved in chronological order" in src
    pdf = build_conversation_pdf(
        transcript=[
            {"role": "customer", "text": "Customer detail " * 80},
            {"role": "agent", "text": "Agent explanation " * 80, "provider": "DeepSeek"},
            {"role": "customer", "text": "Thanks - that makes sense."},
        ],
        profile={"patience": 72, "trust": 75},
        domain="Payment",
        channel="Text Messages",
        session_id="v14-layout",
        satisfaction=87,
        phase="ended",
        language="English",
        analysis="The agent identified the blocker and confirmed the final state.",
    )
    assert pdf.startswith(b"%PDF") and len(pdf) > 2500


def test_v14_toolbar_optical_offsets():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert 'APP_VERSION = "1.4.1-pdf-natural-dialogue"' in source
    assert "transform:translateX(4px)!important" in source
    assert source.count("transform:translateX(4px)!important") >= 2
