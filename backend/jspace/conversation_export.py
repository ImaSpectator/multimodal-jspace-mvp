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


def _text(value: object) -> str:
    """Escape user/model output once and preserve intentional line breaks."""
    return escape(str(value or "")).replace("\n", "<br/>")


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
    """Build a clean transcript report with non-overlapping, full-width turn cards.

    Each customer/agent message is formatted independently before it is placed into a
    single-cell ReportLab table.  There are no nested chat bubbles, alternating x
    offsets, or lists of flowables inside table cells, which avoids the overlap seen
    in earlier exports when long messages wrapped across lines/pages.
    """
    buf = BytesIO()
    font = _font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title="JSpace Live Conversation",
        author="JSpace Live",
    )
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "JTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16324A"),
        spaceAfter=5,
    )
    subtitle = ParagraphStyle(
        "JSubtitle",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=8.6,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#617486"),
        spaceAfter=10,
    )
    section = ParagraphStyle(
        "JSection",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#244B67"),
        spaceBefore=8,
        spaceAfter=7,
    )
    meta_label = ParagraphStyle(
        "JMetaLabel",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#6A7D8C"),
        spaceAfter=1,
    )
    meta_value = ParagraphStyle(
        "JMetaValue",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.0,
        leading=11.2,
        textColor=colors.HexColor("#223645"),
    )
    turn_label = ParagraphStyle(
        "JTurnLabel",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#60798D"),
        spaceAfter=3,
    )
    turn_message = ParagraphStyle(
        "JTurnMessage",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.8,
        leading=14.0,
        textColor=colors.HexColor("#172A38"),
        wordWrap="CJK",
        splitLongWords=True,
    )
    provider_style = ParagraphStyle(
        "JProvider",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=7.1,
        leading=9.2,
        textColor=colors.HexColor("#728595"),
        spaceBefore=3,
    )
    analysis_style = ParagraphStyle(
        "JAnalysis",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.3,
        leading=13.5,
        textColor=colors.HexColor("#243744"),
        wordWrap="CJK",
        spaceAfter=5,
    )

    story: list = [
        Paragraph(_label("JSpace Live Conversation", "JSpace Live 对话记录", language), title),
        Paragraph(
            _label(
                "Customer-service practice transcript",
                "客服练习对话记录",
                language,
            ),
            subtitle,
        ),
    ]

    # A simple metadata grid replaces the old compressed inline header.
    meta_rows = [
        [
            [Paragraph(_label("DOMAIN", "领域", language), meta_label), Paragraph(_text(domain), meta_value)],
            [Paragraph(_label("CHANNEL", "渠道", language), meta_label), Paragraph(_text(channel), meta_value)],
        ],
        [
            [Paragraph(_label("SESSION", "会话", language), meta_label), Paragraph(_text(session_id), meta_value)],
            [Paragraph(_label("PHASE", "阶段", language), meta_label), Paragraph(_text(phase), meta_value)],
        ],
        [
            [Paragraph(_label("PATIENCE", "耐心度", language), meta_label), Paragraph(f"{max(0, int(profile.get('patience', 0)))}/100", meta_value)],
            [Paragraph(_label("TRUST", "信任度", language), meta_label), Paragraph(f"{int(profile.get('trust', 0))}/100", meta_value)],
        ],
        [
            [Paragraph(_label("SATISFACTION", "满意度", language), meta_label), Paragraph(f"{satisfaction:.0f}/100", meta_value)],
            [Paragraph(_label("MESSAGES", "消息数", language), meta_label), Paragraph(str(sum(1 for r in transcript if r.get('role') in {'customer', 'agent'})), meta_value)],
        ],
    ]
    meta_table = Table(meta_rows, colWidths=[doc.width / 2, doc.width / 2], hAlign="LEFT")
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#D8E2EA")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E3EAF0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([meta_table, Spacer(1, 12), Paragraph(_label("Conversation transcript", "对话记录", language), section)])

    turn_number = 0
    for row in transcript:
        role = str(row.get("role") or "customer")
        if role not in {"customer", "agent"}:
            continue
        turn_number += 1
        is_customer = role == "customer"
        who = _label("CUSTOMER", "客户", language) if is_customer else _label("SUPPORT AGENT", "客服", language)
        role_bg = colors.HexColor("#EEF6FF") if is_customer else colors.HexColor("#F4F6F8")
        border = colors.HexColor("#6EA8D9") if is_customer else colors.HexColor("#8193A2")
        role_text = colors.HexColor("#225F91") if is_customer else colors.HexColor("#4F6575")

        # A single composed Paragraph is the only flowable inside the message cell.
        # This makes wrapping deterministic and prevents customer/agent cards from
        # colliding even when model output is long.
        turn_label.textColor = role_text
        label_html = f"<b>{escape(who)}</b> / {_label('Turn', '第', language)} {turn_number}"
        message = _text(row.get("text") or "")
        content = [
            Paragraph(label_html, turn_label),
            Paragraph(message, turn_message),
        ]
        provider = str(row.get("provider") or "").strip().replace("·", "-")
        if provider and not is_customer:
            content.append(Paragraph(f"{_label('Provider', '模型来源', language)}: {_text(provider)}", provider_style))

        card = Table([[content]], colWidths=[doc.width], hAlign="LEFT", splitByRow=1)
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), role_bg),
            ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#D4E0E9")),
            ("LINEBEFORE", (0, 0), (0, 0), 3.0, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 13),
            ("RIGHTPADDING", (0, 0), (-1, -1), 13),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(card)
        story.append(Spacer(1, 7))

    if analysis:
        story.extend([
            Spacer(1, 8),
            Paragraph(_label("Conversation analysis", "对话分析", language), section),
        ])
        analysis_card_content = []
        for block in str(analysis).split("\n"):
            if block.strip():
                analysis_card_content.append(Paragraph(_text(block.strip()), analysis_style))
            else:
                analysis_card_content.append(Spacer(1, 3))
        analysis_card = Table([[analysis_card_content]], colWidths=[doc.width], hAlign="LEFT")
        analysis_card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBFCFD")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8E2EA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(analysis_card)

    doc.build(story)
    return buf.getvalue()
