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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
        "Customer", parent=body, leftIndent=0, rightIndent=0, spaceBefore=0, spaceAfter=0,
        fontSize=9.6, leading=13.8,
    )
    agent_style = ParagraphStyle(
        "Agent", parent=body, leftIndent=0, rightIndent=0, spaceBefore=0, spaceAfter=0,
        fontSize=9.6, leading=13.8,
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

    speaker_style = ParagraphStyle(
        "Speaker", parent=body, fontName=font, fontSize=9.7, leading=12.2,
        textColor=colors.HexColor("#17354D"), spaceBefore=0, spaceAfter=0,
    )
    provider_style = ParagraphStyle(
        "Provider", parent=meta, fontName=font, fontSize=7.4, leading=9.2,
        textColor=colors.HexColor("#65798A"), spaceBefore=0, spaceAfter=0,
    )

    # Render each turn as a single full-width transcript row: fixed speaker column on
    # the left, message content on the right. There is no alternating horizontal
    # offset, so PDF viewers cannot visually stack or overlap customer/agent cards.
    speaker_col = 1.08 * inch
    message_col = doc.width - speaker_col
    for row in transcript:
        role = str(row.get("role") or "customer")
        if role not in {"customer", "agent"}:
            continue
        who = _label("Customer", "客户", language) if role == "customer" else _label("Support Agent", "客服", language)
        message_text = escape(str(row.get("text") or "")).replace("\n", "<br/>")
        provider = str(row.get("provider") or "").strip().replace("·", "-")

        speaker_bg = colors.HexColor("#DCEAF7") if role == "customer" else colors.HexColor("#E7EDF4")
        message_bg = colors.HexColor("#F5F9FD") if role == "customer" else colors.HexColor("#FAFBFD")
        rule = colors.HexColor("#B8CADB") if role == "customer" else colors.HexColor("#CBD5DF")

        message_flow = [Paragraph(message_text, customer_style if role == "customer" else agent_style)]
        if provider and role == "agent":
            message_flow.append(Spacer(1, 3))
            message_flow.append(Paragraph(escape(provider), provider_style))

        turn = Table(
            [[Paragraph(f"<b>{escape(who)}</b>", speaker_style), message_flow]],
            colWidths=[speaker_col, message_col],
            hAlign="LEFT",
            splitByRow=1,
        )
        turn.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), speaker_bg),
            ("BACKGROUND", (1, 0), (1, 0), message_bg),
            ("BOX", (0, 0), (-1, -1), 0.45, rule),
            ("LINEAFTER", (0, 0), (0, 0), 0.45, rule),
            ("LEFTPADDING", (0, 0), (0, 0), 9),
            ("RIGHTPADDING", (0, 0), (0, 0), 8),
            ("TOPPADDING", (0, 0), (0, 0), 10),
            ("BOTTOMPADDING", (0, 0), (0, 0), 10),
            ("LEFTPADDING", (1, 0), (1, 0), 11),
            ("RIGHTPADDING", (1, 0), (1, 0), 11),
            ("TOPPADDING", (1, 0), (1, 0), 9),
            ("BOTTOMPADDING", (1, 0), (1, 0), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(turn)
        story.append(Spacer(1, 8))

    if analysis:
        story.extend([Spacer(1, 10), Paragraph(_label("Conversation analysis", "对话分析", language), h2)])
        for block in str(analysis).split("\n"):
            if block.strip():
                story.append(Paragraph(escape(block.strip()), body))
            else:
                story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()
