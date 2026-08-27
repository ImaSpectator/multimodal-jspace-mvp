from pathlib import Path

from backend.jspace.conversation_export import build_conversation_pdf
from backend.jspace.scenario_generator import generate_manual_context
from backend.jspace.simulator import new_manual_state, update_customer_relationship

ROOT = Path(__file__).parents[1]


def _source():
    return (ROOT / "frontend" / "app.py").read_text()


def test_v12_version_and_header_toolbar_is_higher_and_borderless():
    source = _source()
    assert 'APP_VERSION = "1.2.0-export-analysis"' in source
    assert 'margin-top:-1.15rem' in source
    assert 'border:0!important' in source
    assert 'background:transparent!important' in source
    assert 'font-size:1.14rem!important' in source


def test_v12_suggestion_button_has_per_turn_identity_and_safe_prefill():
    source = _source()
    assert 'key=f"use_manual_suggestion_{sum(1 for r in manual_state.transcript' in source
    assert 'on_click=_queue_manual_suggestion' in source
    assert 'manual_chat_prefill' in source


def test_v12_starting_patience_varies_by_customer_context():
    values = [generate_manual_context("payment", seed=i)[0]["patience"] for i in range(1, 30)]
    assert len(set(values)) > 4
    assert all(28 <= v <= 100 for v in values)
    assert any(v < 80 for v in values)


def test_v12_relationship_update_can_drop_below_zero():
    profile, backend = generate_manual_context("account_access", seed=9)
    profile["patience"] = 2
    state = new_manual_state(capacity_k=4, backend_events=backend, profile=profile)
    state.conflicts = [object()]  # only truthiness is used in relationship update
    state.transcript.extend([{"role": "agent", "text": "x"}] * 5)
    update_customer_relationship(profile, state, "Please try again and repeat the reset again.", "Local fallback")
    assert profile["patience"] < 0


def test_v12_manual_auto_end_and_post_session_actions_exist():
    source = _source()
    assert 'if float(profile.get("patience", 0)) < 0:' in source
    assert 'state.session_ended = True' in source
    assert 'def render_post_session_actions' in source
    assert 'Analyze conversation' in source
    assert 'Save conversation as PDF' in source


def test_v12_pdf_export_generates_valid_english_and_chinese_pdf():
    transcript = [
        {"role": "customer", "text": "My account is locked."},
        {"role": "agent", "text": "I found the blocker and removed it.", "provider": "DeepSeek"},
    ]
    profile = {"patience": 64, "trust": 71}
    en = build_conversation_pdf(
        transcript=transcript, profile=profile, domain="Account access", channel="Text Messages",
        session_id="test-en", satisfaction=82, phase="ended", language="English", analysis="Resolved cleanly.",
    )
    zh = build_conversation_pdf(
        transcript=[{"role": "customer", "text": "我的账户被锁定了。"}, {"role": "agent", "text": "已经找到阻塞原因并处理。"}],
        profile=profile, domain="账户访问", channel="文字消息", session_id="test-zh",
        satisfaction=82, phase="ended", language="Simplified Chinese", analysis="问题已经解决。",
    )
    assert en.startswith(b"%PDF") and len(en) > 1500
    assert zh.startswith(b"%PDF") and len(zh) > 1500


def test_v12_analysis_is_button_triggered_not_automatic():
    source = _source()
    assert 'if st.button(L("Analyze conversation", "分析对话")' in source
    assert 'analyze_conversation_summary(' in source
