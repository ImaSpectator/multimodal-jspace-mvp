from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).parents[1]


def _load_in_app_renderer():
    source = (ROOT / "frontend" / "app.py").read_text()
    start = source.index("# Canonical website PDF renderer.")
    end = source.index("\ndef update_customer_relationship", start)
    block = source[start:end]
    prelude = """
import re
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
"""
    ns = {}
    exec(prelude + "\n" + block, ns, ns)
    return ns["build_website_plain_transcript_pdf"]


def _kwargs(language="English"):
    return dict(
        transcript=[
            {"role": "customer", "text": "The app still shows the old room.", "emotion": "frustrated", "emotion_intensity": 0.8, "nonverbal_cue": "text-derived affect"},
            {"role": "agent", "text": "I corrected the reservation record.", "provider": "DeepSeek - model"},
            {"role": "customer", "text": "Great, thank you."},
            {"role": "agent", "text": "You're all set.", "provider": "DeepSeek - model"},
        ],
        profile={"patience": 70, "trust": 80},
        domain="Hotel Hospitality" if language == "English" else "酒店与住宿",
        channel="Text Messages" if language == "English" else "文字消息",
        session_id="website-inline-test",
        satisfaction=90,
        phase="ended",
        language=language,
        analysis="**Summary**\nThe issue was resolved." if language == "English" else "**总结**\n问题已经解决。",
    )


def test_v146_website_pdf_button_calls_in_app_renderer_not_backend_renderer():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert "pdf_bytes = build_website_plain_transcript_pdf(" in source
    assert "from backend.jspace.conversation_export import" not in source
    assert 'key=f"download_pdf_plain_v146_{state.session_id}"' in source


def test_v146_in_app_renderer_outputs_approved_plain_transcript_structure():
    build = _load_in_app_renderer()
    data = build(**_kwargs())
    assert data.startswith(b"%PDF")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    assert "Conversation" in text
    assert "Turn 01" in text
    assert "Customer" in text
    assert "The app still shows the old room." in text
    assert "Support Agent" in text
    assert "I corrected the reservation record." in text
    assert "Conversation Analysis" in text
    assert "Summary" in text


def test_v146_in_app_renderer_outputs_chinese_labels_and_text():
    build = _load_in_app_renderer()
    data = build(**_kwargs("Chinese"))
    assert data.startswith(b"%PDF")
    reader = PdfReader(BytesIO(data))
    assert len(reader.pages) >= 2
    source = (ROOT / "frontend" / "app.py").read_text()
    for label in ["JSpace Live 对话记录", "对话", "轮次", "客户", "客服", "对话分析"]:
        assert label in source
