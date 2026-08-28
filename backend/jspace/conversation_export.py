from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def _inline_markdown(value: str) -> str:
    """Render the small amount of markdown produced by the conversation analysis."""
    safe = escape(str(value or ""))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe)


def _analysis_items(analysis: str) -> list[tuple[str, str]]:
    """Convert loose markdown into readable heading/body/bullet blocks.

    Model summaries and copied PDFs may wrap one logical bullet across several text
    lines.  Continuation lines stay attached to the current bullet/body so the export
    reads like a report rather than a line-by-line transcript dump.
    """
    items: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_kind, current_parts
        if current_kind and current_parts:
            items.append((current_kind, " ".join(x.strip() for x in current_parts if x.strip())))
        current_kind = None
        current_parts = []

    for raw in str(analysis or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if re.fullmatch(r"\*\*.+\*\*", line):
            flush()
            items.append(("heading", line[2:-2].strip()))
            continue
        if line.startswith("- "):
            flush()
            current_kind = "bullet"
            current_parts = [line[2:].strip()]
            continue
        if current_kind == "bullet":
            current_parts.append(line)
        else:
            if current_kind != "body":
                flush()
                current_kind = "body"
            current_parts.append(line)
    flush()
    return items


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
    """Create a professional, archive-friendly conversation report.

    The export deliberately does *not* imitate the compact in-app chat bubbles.  PDF is
    a record people may save, print, or review later, so the layout uses normal report
    typography, generous whitespace, page numbers, session metadata, readable metrics,
    and one independently splittable paragraph card per message.  This avoids the
    overlap/smushing problems caused by nested flowables and offset chat bubbles.
    """
    buf = BytesIO()
    font = _font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.62 * inch,
        title=_label("JSpace Live Conversation Report", "JSpace Live 对话报告", language),
        author="JSpace Live",
        subject=_label("Customer-service conversation record", "客服对话记录", language),
    )
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "JTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=21,
        leading=25,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#173A54"),
        spaceAfter=3,
    )
    subtitle = ParagraphStyle(
        "JSubtitle",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.2,
        leading=12.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#6C7E8C"),
        spaceAfter=15,
    )
    section = ParagraphStyle(
        "JSection",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#234B67"),
        spaceBefore=10,
        spaceAfter=8,
    )
    small_label = ParagraphStyle(
        "JSmallLabel",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#718392"),
    )
    meta_value = ParagraphStyle(
        "JMetaValue",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.7,
        leading=12.4,
        textColor=colors.HexColor("#203645"),
    )
    metric = ParagraphStyle(
        "JMetric",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#213847"),
    )

    customer_card = ParagraphStyle(
        "JCustomerCard",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=10.6,
        leading=15.2,
        textColor=colors.HexColor("#162E3F"),
        backColor=colors.HexColor("#F1F7FD"),
        borderColor=colors.HexColor("#B9D4E8"),
        borderWidth=0.65,
        borderPadding=(10, 12, 11, 12),
        borderRadius=4,
        spaceAfter=9,
        wordWrap="CJK",
        splitLongWords=True,
    )
    agent_card = ParagraphStyle(
        "JAgentCard",
        parent=customer_card,
        textColor=colors.HexColor("#1D303C"),
        backColor=colors.HexColor("#F7F8FA"),
        borderColor=colors.HexColor("#D2D9DF"),
    )
    analysis_heading = ParagraphStyle(
        "JAnalysisHeading",
        parent=styles["Heading3"],
        fontName=font,
        fontSize=11.4,
        leading=14.5,
        textColor=colors.HexColor("#244B67"),
        spaceBefore=7,
        spaceAfter=4,
    )
    analysis_body = ParagraphStyle(
        "JAnalysisBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=10.1,
        leading=14.7,
        textColor=colors.HexColor("#263A47"),
        spaceAfter=6,
        wordWrap="CJK",
    )
    analysis_bullet = ParagraphStyle(
        "JAnalysisBullet",
        parent=analysis_body,
        leftIndent=15,
        firstLineIndent=-8,
        bulletIndent=4,
        spaceAfter=5,
    )
    note = ParagraphStyle(
        "JNote",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#728391"),
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message_count = sum(1 for r in transcript if r.get("role") in {"customer", "agent"})
    customer_count = sum(1 for r in transcript if r.get("role") == "customer")

    story: list = [
        Paragraph(_label("JSpace Live Conversation Report", "JSpace Live 对话报告", language), title),
        Paragraph(
            _label(
                "A structured record of the customer interaction, outcome, and post-session analysis.",
                "客户互动、处理结果与会后分析的结构化记录。",
                language,
            ),
            subtitle,
        ),
        Paragraph(_label("Session details", "会话信息", language), section),
    ]

    def meta_cell(label: str, value: object) -> Paragraph:
        return Paragraph(
            f'<font size="7" color="#718392"><b>{escape(label.upper())}</b></font><br/>{_text(value)}',
            meta_value,
        )

    details = [
        [
            meta_cell(_label("Domain", "领域", language), domain),
            meta_cell(_label("Channel", "渠道", language), channel),
        ],
        [
            meta_cell(_label("Session ID", "会话 ID", language), session_id),
            meta_cell(_label("Final phase", "最终阶段", language), phase),
        ],
        [
            meta_cell(_label("Conversation turns", "客户轮次", language), customer_count),
            meta_cell(_label("Report generated", "报告生成时间", language), generated_at),
        ],
    ]
    details_table = Table(details, colWidths=[doc.width / 2, doc.width / 2], hAlign="LEFT")
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFC")),
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#D8E1E8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E7ECF0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([details_table, Spacer(1, 10)])

    def metric_cell(label: str, value: str) -> Paragraph:
        return Paragraph(
            f'<font size="7" color="#718392"><b>{escape(label.upper())}</b></font><br/>'
            f'<font size="16" color="#173A54"><b>{escape(value)}</b></font>',
            metric,
        )

    metrics = Table([[ 
        metric_cell(_label("Satisfaction", "满意度", language), f"{satisfaction:.0f}/100"),
        metric_cell(_label("Trust", "信任度", language), f"{int(profile.get('trust', 0))}/100"),
        metric_cell(_label("Patience", "耐心度", language), f"{max(0, int(profile.get('patience', 0)))}/100"),
    ]], colWidths=[doc.width / 3] * 3, hAlign="LEFT")
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FB")),
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#D5E2EB")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DFE8EE")),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([
        metrics,
        Spacer(1, 15),
        Paragraph(_label("Conversation transcript", "对话记录", language), section),
        Paragraph(
            _label(
                "Customer and support messages are preserved in chronological order. Agent model/provider details appear in smaller text for traceability.",
                "客户与客服消息按时间顺序保留。客服模型/来源以较小文字显示，便于追溯。",
                language,
            ),
            note,
        ),
        Spacer(1, 6),
    ])

    exchange = 0
    for row in transcript:
        role = str(row.get("role") or "customer")
        if role not in {"customer", "agent"}:
            continue
        is_customer = role == "customer"
        if is_customer:
            exchange += 1
        shown_exchange = max(1, exchange)
        who = _label("Customer", "客户", language) if is_customer else _label("Support agent", "客服", language)
        turn_word = _label("Turn", "轮次", language) if is_customer else _label("Response", "回复", language)
        role_color = "#25689B" if is_customer else "#4D6372"
        label_html = (
            f'<font size="8" color="{role_color}"><b>{escape(who.upper())}</b></font>'
            f'<font size="8" color="#8293A0"> | {escape(turn_word)} {shown_exchange:02d}</font>'
        )
        message_html = _text(row.get("text") or "")
        provider = str(row.get("provider") or "").strip().replace("·", "-")
        provider_html = ""
        if provider and not is_customer:
            provider_html = (
                f'<br/><br/><font size="7.5" color="#7D8D99">'
                f'{escape(_label("Provider", "模型来源", language))}: {_text(provider)}</font>'
            )
        story.append(Paragraph(f"{label_html}<br/><br/>{message_html}{provider_html}", customer_card if is_customer else agent_card))
        story.append(Spacer(1, 5 if is_customer else 11))

    if analysis:
        # Put analysis on a fresh page so the transcript never gets squeezed to make
        # room for a dense post-session model summary.
        story.extend([
            PageBreak(),
            Paragraph(_label("Conversation analysis", "对话分析", language), title),
            Paragraph(
                _label(
                    "Post-session observations are formatted as a review document rather than raw markdown.",
                    "会后观察以评审文档格式呈现，而不是原始 Markdown 文本。",
                    language,
                ),
                subtitle,
            ),
        ])
        for kind, text in _analysis_items(analysis):
            if kind == "heading":
                story.append(Paragraph(_inline_markdown(text), analysis_heading))
            elif kind == "bullet":
                story.append(Paragraph(_inline_markdown(text), analysis_bullet, bulletText="•"))
            else:
                story.append(Paragraph(_inline_markdown(text), analysis_body))

    def decorate_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#DFE6EB"))
        canvas.setLineWidth(0.45)
        canvas.line(document.leftMargin, 0.48 * inch, letter[0] - document.rightMargin, 0.48 * inch)
        canvas.setFont(font, 7.2)
        canvas.setFillColor(colors.HexColor("#7C8C98"))
        canvas.drawString(document.leftMargin, 0.30 * inch, f"JSpace Live  |  {session_id}")
        page_label = _label("Page", "第", language)
        canvas.drawRightString(letter[0] - document.rightMargin, 0.30 * inch, f"{page_label} {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buf.getvalue()
