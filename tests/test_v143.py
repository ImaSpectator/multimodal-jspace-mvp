from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from backend.jspace.conversation_export import build_conversation_pdf
from backend.jspace.engine import detect_conflicts
from backend.jspace.schemas import Concept

ROOT = Path(__file__).parents[1]


def _concept(name: str, value: str) -> Concept:
    return Concept(
        id=f"{name}-1",
        name=name,
        value=value,
        sources=["derived"],
        confidence=0.9,
        task_relevance=0.9,
    )


def test_v143_pdf_is_plain_text_transcript_with_role_labels_in_reading_order():
    transcript = [
        {
            "role": "customer",
            "text": "My upgrade is still showing the old room type.",
            "emotion": "frustrated",
            "emotion_intensity": 0.8,
            "nonverbal_cue": "text-derived affect",
        },
        {
            "role": "agent",
            "text": "I found the sync issue and corrected the reservation record.",
            "provider": "DeepSeek · deepseek/deepseek-v4-flash-vision-exp",
        },
        {"role": "customer", "text": "Great, thank you. That's all I needed."},
        {"role": "agent", "text": "You're all set. Enjoy your stay!", "provider": "DeepSeek"},
    ]
    pdf = build_conversation_pdf(
        transcript=transcript,
        profile={"patience": 70, "trust": 75},
        domain="Hotel Hospitality",
        channel="Text Messages",
        session_id="plain-transcript-test",
        satisfaction=88,
        phase="ended",
        language="English",
        analysis="**Summary and outcome**\nThe reservation sync issue was corrected.\n- The final state was confirmed.",
    )
    assert pdf.startswith(b"%PDF")

    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)
    customer_i = extracted.index("Customer")
    customer_text_i = extracted.index("My upgrade is still showing the old room type.")
    agent_i = extracted.index("Support Agent")
    agent_text_i = extracted.index("I found the sync issue and corrected the reservation record.")
    assert customer_i < customer_text_i < agent_i < agent_text_i
    assert "Conversation Analysis" in extracted
    assert "The reservation sync issue was corrected." in extracted

    source = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert "Table(" not in source
    assert "chat bubbles, cards, or tables" in source


def test_v143_high_conflict_has_fully_chinese_explanation():
    conflicts = detect_conflicts([
        _concept("authoritative_status", "unresolved"),
        _concept("customer_visible_status", "connected"),
    ])
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.severity == "high"
    assert conflict.description_zh == "面向客户的证据显示“已连接”，但权威系统仍显示问题尚未解决。"
    assert "connected" not in conflict.description_zh


def test_v143_medium_conflict_has_fully_chinese_explanation():
    intensity = _concept("emotion_intensity", "0.80")
    conflicts = detect_conflicts([
        _concept("customer_belief_status", "resolved"),
        _concept("customer_emotion", "frustrated"),
        intensity,
    ])
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.severity == "medium"
    assert conflict.description_zh == "客户表示问题已经解决，但其“沮丧”情绪仍然较强（80%）。"
    assert "frustrated" not in conflict.description_zh


def test_v143_workspace_uses_localized_conflict_explanation_not_raw_english():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert "def display_conflict_description(conflict)" in source
    assert 'getattr(conflict, "description_zh", "")' in source
    assert "html.escape(display_conflict_description(conflict))" in source
