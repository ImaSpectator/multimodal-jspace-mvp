from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ReportLab ships CID font support; no external font file is bundled or exposed.
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception:
    pass


def _font(language: str) -> str:
    return "STSong-Light" if str(language).lower().startswith(("zh", "chinese", "simplified")) else "Helvetica"


def _label(en: str, zh: str, language: str) -> str:
    return zh if _font(language) == "STSong-Light" else en


def build_conversation_pdf(
    *,
    transcript: list[dict],
    profile: dict,
    domain: str,
    channel: str,
    session_id: str,
    satisfaction: float,
    phase: str,
    language: str = "English",
    analysis: str | None = None,
) -> bytes:
    """Create a compact, downloadable conversation report PDF entirely in memory."""
    buf = BytesIO()
    font = _font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title="JSpace Live Conversation",
        author="JSpace Live",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "JTitle", parent=styles["Title"], fontName=font, fontSize=19, leading=23,
        alignment=TA_CENTER, textColor=colors.HexColor("#16324A"), spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "JH2", parent=styles["Heading2"], fontName=font, fontSize=12.5, leading=16,
        textColor=colors.HexColor("#244B67"), spaceBefore=10, spaceAfter=6,
    )
    body = ParagraphStyle(
        "JBody", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=13.2,
        textColor=colors.HexColor("#202B35"), spaceAfter=5, wordWrap="CJK",
    )
    meta = ParagraphStyle(
        "JMeta", parent=body, fontSize=8.6, leading=12, textColor=colors.HexColor("#5A6D7D"),
    )
    customer_style = ParagraphStyle(
        "Customer", parent=body, leftIndent=22, rightIndent=4, borderColor=colors.HexColor("#B7CBE0"),
        borderWidth=0.5, borderPadding=7, backColor=colors.HexColor("#F1F6FB"), spaceBefore=4, spaceAfter=7,
    )
    agent_style = ParagraphStyle(
        "Agent", parent=body, leftIndent=4, rightIndent=22, borderColor=colors.HexColor("#C9D4E2"),
        borderWidth=0.5, borderPadding=7, backColor=colors.HexColor("#F7F9FC"), spaceBefore=4, spaceAfter=7,
    )

    story = [
        Paragraph(_label("JSpace Live Conversation", "JSpace Live 对话记录", language), title),
        Paragraph(
            escape(
                f"{_label('Domain', '领域', language)}: {domain}   |   "
                f"{_label('Channel', '渠道', language)}: {channel}   |   "
                f"{_label('Session', '会话', language)}: {session_id}"
            ), meta,
        ),
        Paragraph(
            escape(
                f"{_label('Patience', '耐心度', language)}: {max(0, int(profile.get('patience', 0)))}/100   |   "
                f"{_label('Trust', '信任度', language)}: {int(profile.get('trust', 0))}/100   |   "
                f"{_label('Satisfaction', '满意度', language)}: {satisfaction:.0f}/100   |   "
                f"{_label('Phase', '阶段', language)}: {phase}"
            ), meta,
        ),
        Spacer(1, 8),
        Paragraph(_label("Conversation", "对话", language), h2),
    ]

    for row in transcript:
        role = str(row.get("role") or "customer")
        if role not in {"customer", "agent"}:
            continue
        who = _label("Customer", "客户", language) if role == "customer" else _label("Support Agent", "客服", language)
        text = escape(str(row.get("text") or "")).replace("\n", "<br/>")
        provider = str(row.get("provider") or "").strip()
        provider_html = f"<br/><font size='7' color='#6A7E90'>{escape(provider)}</font>" if provider and role == "agent" else ""
        story.append(Paragraph(f"<b>{escape(who)}</b><br/>{text}{provider_html}", customer_style if role == "customer" else agent_style))

    if analysis:
        story.extend([Spacer(1, 10), Paragraph(_label("Conversation analysis", "对话分析", language), h2)])
        for block in str(analysis).split("\n"):
            if block.strip():
                story.append(Paragraph(escape(block.strip()), body))
            else:
                story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()
