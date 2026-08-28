from pathlib import Path

from backend.jspace.conversation_export import build_conversation_pdf
from backend.jspace.scenario_generator import generate_manual_context
from backend.jspace.simulator import apply_manual_customer_message, append_agent_reply, new_manual_state

ROOT = Path(__file__).parents[1]


def _source():
    return (ROOT / "frontend" / "app.py").read_text()


def test_v13_toolbar_is_fixed_borderless_and_top_right():
    source = _source()
    assert 'APP_VERSION = "1.4.0-natural-manual-pdf"' in source
    assert 'position:fixed!important' not in source
    assert 'st.columns([12.0, 1.15, .72, .72, .72, .72]' in source
    assert 'border:0!important' in source
    assert 'background:transparent!important' in source
    assert ':material/settings:' in source and ':material/help:' in source and ':material/link:' in source
    assert 'transform:translateX(4px)!important' in source
    assert 'transform:translateX(6px)!important' in source


def test_v13_suggestions_have_unused_move_and_closing_logic():
    source = _source()
    assert 'def _unused_customer_move' in source
    assert 'if _manual_ready_to_close(state):' in source
    assert "Everything looks good now. That's all I needed" in source
    assert 'anything else' in source and 'other questions' in source


def test_v13_manual_closing_turn_ends_after_agent_goodbye():
    profile, backend = generate_manual_context("account_access", seed=2)
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    state.conflicts = []
    state.customer_satisfaction = 88
    apply_manual_customer_message(state, "Everything looks good now. That's all I needed - thank you and have a good day!")
    assert state.session_phase == "closing"
    append_agent_reply(state, "You're all set. Thanks for contacting support, and have a great day!", "DeepSeek")
    assert state.session_ended is True
    assert state.session_phase == "ended"


def test_v13_pdf_uses_full_width_separate_message_cards():
    source = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert 'single-cell ReportLab table' in source
    assert 'colWidths=[doc.width]' in source
    assert '("LINEBEFORE", (0, 0), (0, 0), 3.0, border)' in source
    assert 'nested chat bubbles' in source
    pdf = build_conversation_pdf(
        transcript=[
            {"role":"customer","text":"This is a longer customer message that should wrap safely without touching the next message. " * 4},
            {"role":"agent","text":"This is a longer support response that should occupy its own bubble and remain separated from the following row. " * 4,"provider":"DeepSeek"},
            {"role":"customer","text":"Thanks, that resolves it."},
        ],
        profile={"patience":70,"trust":72}, domain="Account access", channel="Text Messages",
        session_id="v13", satisfaction=90, phase="ended", language="English", analysis="Resolved successfully.",
    )
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000
