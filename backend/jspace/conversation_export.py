from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Built-in CID font support keeps Chinese exports self-contained without bundling fonts.
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception:
    pass


NAVY = colors.HexColor("#173A54")
BLUE = colors.HexColor("#2F6F9F")
INK = colors.HexColor("#20313D")
MUTED = colors.HexColor("#6E7F8C")
LINE = colors.HexColor("#DCE5EB")
PANEL = colors.HexColor("#F7F9FB")
CUSTOMER_BG = colors.HexColor("#EEF6FC")
AGENT_BG = colors.HexColor("#FFFFFF")
SUCCESS_BG = colors.HexColor("#EEF8F3")
SUCCESS = colors.HexColor("#287A55")
WARNING_BG = colors.HexColor("#FFF7E8")
WARNING = colors.HexColor("#9B6A16")



EMOTION_ZH = {
    "calm": "平静", "neutral": "中性", "curious": "好奇", "hopeful": "有希望",
    "appreciative": "感谢", "satisfied": "满意", "relieved": "放松", "uncertain": "不确定",
    "confused": "困惑", "anxious": "焦虑", "disappointed": "失望", "frustrated": "沮丧",
    "angry": "生气", "impatient": "不耐烦", "skeptical": "怀疑", "distressed": "紧张",
    "embarrassed": "尴尬",
}


def _display_emotion(emotion: str, language: str) -> str:
    key = str(emotion or "").strip().lower()
    if not key:
        return ""
    if _font(language) == "STSong-Light":
        return EMOTION_ZH.get(key, key)
    return key.replace("_", " ").title()


def _display_affect_source(value: str, language: str) -> str:
    text = str(value or "").strip()
    if _font(language) != "STSong-Light":
        return text
    low = text.lower()
    if "text-derived" in low or "inferred from text" in low:
        return "文字推断情绪"
    if "audio-derived" in low or "inferred from audio" in low:
        return "音频转写文字推断情绪"
    if "video-derived" in low or "inferred from video" in low:
        return "视频内容推断情绪"
    return text

def _font(language: str) -> str:
    return "STSong-Light" if str(language).lower().startswith(("zh", "chinese", "simplified")) else "Helvetica"


def _label(en: str, zh: str, language: str) -> str:
    return zh if _font(language) == "STSong-Light" else en


def _text(value: object) -> str:
    return escape(str(value or "")).replace("\n", "<br/>")


def _inline_markdown(value: str) -> str:
    safe = escape(str(value or ""))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe)


def _analysis_items(analysis: str) -> list[tuple[str, str]]:
    """Turn loose model markdown into stable report blocks."""
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
        heading = re.fullmatch(r"\*\*(.+?)\*\*", line)
        if heading:
            flush()
            items.append(("heading", heading.group(1).strip()))
            continue
        if line.startswith(('- ', '• ')):
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


def _phase_display(phase: str, language: str) -> str:
    low = str(phase or "").lower()
    if low in {"ended", "closing", "resolved"}:
        return _label("Resolved / closed", "已解决 / 已结束", language)
    if low == "resolving":
        return _label("Resolution in progress", "处理中", language)
    return _label("Open / incomplete", "未完成", language)


def _final_outcome(transcript: list[dict], phase: str, language: str) -> str:
    """Return the substantive resolution statement, not a generic goodbye."""
    agent_rows = [
        str(row.get("text") or "").strip()
        for row in transcript
        if row.get("role") == "agent" and str(row.get("text") or "").strip()
    ]
    resolution_tokens = (
        "resolved", "fixed", "corrected", "restored", "confirmed", "all set",
        "已解决", "已修复", "已更正", "已恢复", "已确认", "处理完成",
    )
    for text in reversed(agent_rows):
        if any(token in text.lower() for token in resolution_tokens):
            return text
    if agent_rows:
        return agent_rows[-1]
    if str(phase).lower() in {"ended", "closing", "resolved"}:
        return _label("The interaction ended after resolution was confirmed.", "问题确认解决后，对话已结束。", language)
    return _label("The interaction ended without a recorded final agent response.", "对话结束时未记录最终客服回复。", language)


def _provider_summary(transcript: list[dict]) -> str:
    seen: list[str] = []
    for row in transcript:
        if row.get("role") != "agent":
            continue
        provider = str(row.get("provider") or "").strip().replace("·", "-")
        if provider and provider not in seen:
            seen.append(provider)
    return "; ".join(seen) if seen else "-"


def _group_exchanges(transcript: list[dict]) -> list[dict]:
    exchanges: list[dict] = []
    current: dict | None = None
    for row in transcript:
        role = str(row.get("role") or "")
        if role == "customer":
            if current:
                exchanges.append(current)
            current = {"customer": row, "agent": None}
        elif role == "agent":
            if current is None:
                current = {"customer": None, "agent": row}
            elif current.get("agent") is None:
                current["agent"] = row
            else:
                exchanges.append(current)
                current = {"customer": None, "agent": row}
    if current:
        exchanges.append(current)
    return exchanges


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
    """Build an archive-quality conversation record.

    The PDF is intentionally *not* a screenshot-like chat export. It is a business
    record: a concise case overview, clear session metadata, one self-contained block
    per customer/agent exchange, and a separate post-session review. This gives every
    message predictable width and vertical flow, eliminating the overlapping/smushed
    bubble layout that previous exports produced.
    """
    buf = BytesIO()
    font = _font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.70 * inch,
        bottomMargin=0.65 * inch,
        title=_label("JSpace Live Conversation Record", "JSpace Live 对话记录", language),
        author="JSpace Live",
        subject=_label("Customer-service conversation and analysis record", "客服对话与分析记录", language),
    )
    styles = getSampleStyleSheet()

    eyebrow = ParagraphStyle(
        "JEyebrow", parent=styles["BodyText"], fontName=font, fontSize=7.4, leading=9,
        textColor=BLUE, spaceAfter=4,
    )
    title = ParagraphStyle(
        "JTitle", parent=styles["Title"], fontName=font, fontSize=23, leading=27,
        alignment=TA_LEFT, textColor=NAVY, spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "JSubtitle", parent=styles["BodyText"], fontName=font, fontSize=9.4, leading=13,
        textColor=MUTED, spaceAfter=13,
    )
    section = ParagraphStyle(
        "JSection", parent=styles["Heading2"], fontName=font, fontSize=14, leading=17,
        textColor=NAVY, spaceBefore=13, spaceAfter=7,
    )
    section_note = ParagraphStyle(
        "JSectionNote", parent=styles["BodyText"], fontName=font, fontSize=8.3, leading=11.4,
        textColor=MUTED, spaceAfter=8,
    )
    body = ParagraphStyle(
        "JBody", parent=styles["BodyText"], fontName=font, fontSize=10.2, leading=14.6,
        textColor=INK, wordWrap="CJK", splitLongWords=True,
    )
    body_small = ParagraphStyle(
        "JBodySmall", parent=body, fontSize=8.4, leading=11.2, textColor=MUTED,
    )
    meta_label = ParagraphStyle(
        "JMetaLabel", parent=body_small, fontSize=6.9, leading=8, textColor=MUTED,
    )
    meta_value = ParagraphStyle(
        "JMetaValue", parent=body, fontSize=9.2, leading=12.1, textColor=INK,
    )
    metric_style = ParagraphStyle(
        "JMetric", parent=body, fontSize=10, leading=12, alignment=TA_CENTER,
    )
    turn_no = ParagraphStyle(
        "JTurnNo", parent=body_small, fontSize=7.2, leading=9, textColor=BLUE,
    )
    role_label = ParagraphStyle(
        "JRole", parent=body_small, fontSize=7.4, leading=9.4, textColor=BLUE,
    )
    agent_role = ParagraphStyle(
        "JAgentRole", parent=role_label, textColor=colors.HexColor("#596D7B"),
    )
    analysis_heading = ParagraphStyle(
        "JAnalysisHeading", parent=styles["Heading3"], fontName=font, fontSize=11.8,
        leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=4,
    )
    analysis_body = ParagraphStyle(
        "JAnalysisBody", parent=body, fontSize=10, leading=14.5, spaceAfter=6,
    )
    analysis_bullet = ParagraphStyle(
        "JAnalysisBullet", parent=analysis_body, leftIndent=16, firstLineIndent=-8,
        bulletIndent=4, spaceAfter=5,
    )
    footer_style = ParagraphStyle(
        "JFooter", parent=body_small, fontSize=7.2, leading=9, textColor=MUTED,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exchanges = _group_exchanges(transcript)
    first_customer = next(
        (str(row.get("text") or "").strip() for row in transcript if row.get("role") == "customer"),
        _label("No customer message recorded.", "未记录客户消息。", language),
    )
    outcome = _final_outcome(transcript, phase, language)
    trust = int(round(float(profile.get("trust", 0) or 0)))
    patience = max(0, int(round(float(profile.get("patience", 0) or 0))))

    story: list = [
        Paragraph(_label("JSPACE LIVE / CONVERSATION RECORD", "JSPACE LIVE / 对话记录", language), eyebrow),
        Paragraph(_label("Customer Support Interaction Report", "客户支持互动报告", language), title),
        Paragraph(
            _label(
                "A structured record of the customer issue, support actions, resolution status, experience signals, and post-session review.",
                "用于长期保存的客户问题、客服处理过程、解决状态、体验指标与会后评审记录。",
                language,
            ),
            subtitle,
        ),
    ]

    # Case-at-a-glance panel: useful even when someone opens the PDF months later.
    case_panel = Table([
        [Paragraph(f"<b>{escape(_label('Customer issue', '客户问题', language))}</b>", meta_value)],
        [Paragraph(_text(first_customer), body)],
        [Spacer(1, 2)],
        [Paragraph(f"<b>{escape(_label('Resolution / outcome', '解决结果', language))}</b>", meta_value)],
        [Paragraph(_text(outcome), body)],
    ], colWidths=[doc.width])
    case_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 11),
    ]))
    story.extend([case_panel, Spacer(1, 13), Paragraph(_label("Session overview", "会话概览", language), section)])

    def meta_cell(label: str, value: object) -> list[Paragraph]:
        return [
            Paragraph(escape(label.upper()), meta_label),
            Paragraph(_text(value), meta_value),
        ]

    overview = Table([
        [meta_cell(_label("Domain", "领域", language), domain), meta_cell(_label("Channel", "渠道", language), channel)],
        [meta_cell(_label("Session ID", "会话 ID", language), session_id), meta_cell(_label("Status", "状态", language), _phase_display(phase, language))],
        [meta_cell(_label("Customer turns", "客户轮次", language), len(exchanges)), meta_cell(_label("Generated", "生成时间", language), generated_at)],
    ], colWidths=[doc.width / 2, doc.width / 2])
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([overview, Spacer(1, 9)])

    def metric_cell(label: str, value: float, *, is_status: bool = False) -> Paragraph:
        if is_status:
            shown = str(value)
        else:
            shown = f"{float(value):.0f}/100"
        return Paragraph(
            f'<font size="7" color="#6E7F8C">{escape(label.upper())}</font><br/>'
            f'<font size="16" color="#173A54"><b>{escape(shown)}</b></font>',
            metric_style,
        )

    metrics = Table([[
        metric_cell(_label("Satisfaction", "满意度", language), satisfaction),
        metric_cell(_label("Trust", "信任度", language), trust),
        metric_cell(_label("Patience", "耐心度", language), patience),
    ]], colWidths=[doc.width / 3] * 3)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7FA")),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([metrics, Spacer(1, 15), Paragraph(_label("Conversation transcript", "对话记录", language), section)])
    story.append(Paragraph(
        _label(
            "Each turn is grouped as one customer-support exchange. Customer affect and response-provider information are retained as traceability metadata, but kept visually secondary to the conversation itself.",
            "每轮客户与客服回复作为一个完整交流单元保存。客户情绪与模型来源作为追溯元数据保留，但不会干扰正文阅读。",
            language,
        ),
        section_note,
    ))

    for idx, exchange in enumerate(exchanges, 1):
        customer = exchange.get("customer") or {}
        agent = exchange.get("agent") or {}
        customer_text = str(customer.get("text") or "").strip()
        agent_text = str(agent.get("text") or "").strip()
        emotion = str(customer.get("emotion") or "").strip()
        intensity = customer.get("emotion_intensity")
        affect_source = str(customer.get("nonverbal_cue") or "").strip()
        provider = str(agent.get("provider") or "").strip().replace("·", "-")

        customer_meta_bits: list[str] = []
        if emotion:
            customer_meta_bits.append(_display_emotion(emotion, language))
        if isinstance(intensity, (int, float)):
            customer_meta_bits.append(f"{float(intensity):.0%}")
        if affect_source:
            customer_meta_bits.append(_display_affect_source(affect_source, language))
        customer_meta = "  |  ".join(customer_meta_bits)

        rows: list[list] = [
            [Paragraph(f"{escape(_label('TURN', '轮次', language))} {idx:02d}", turn_no)],
            [Paragraph(f"<b>{escape(_label('Customer', '客户', language))}</b>", role_label)],
            [Paragraph(_text(customer_text) if customer_text else "-", body)],
        ]
        if customer_meta:
            rows.append([Paragraph(_text(customer_meta), body_small)])
        rows.extend([
            [Paragraph(f"<b>{escape(_label('Support agent', '客服', language))}</b>", agent_role)],
            [Paragraph(_text(agent_text) if agent_text else _label("No response recorded.", "未记录回复。", language), body)],
        ])
        if provider:
            rows.append([Paragraph(f"{escape(_label('Response source', '回复来源', language))}: {_text(provider)}", body_small)])

        turn_table = Table(rows, colWidths=[doc.width], splitByRow=1)
        # Backgrounds are applied by semantic row rather than alternating chat bubbles.
        customer_end = 3 if not customer_meta else 4
        agent_start = customer_end
        turn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PANEL),
            ("BACKGROUND", (0, 1), (-1, customer_end - 1), CUSTOMER_BG),
            ("BACKGROUND", (0, agent_start), (-1, -1), AGENT_BG),
            ("BOX", (0, 0), (-1, -1), 0.65, LINE),
            ("LINEBEFORE", (0, 0), (0, -1), 3.5, BLUE),
            ("LINEABOVE", (0, agent_start), (-1, agent_start), 0.55, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 13),
            ("RIGHTPADDING", (0, 0), (-1, -1), 13),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
            ("TOPPADDING", (0, agent_start), (-1, agent_start), 8),
        ]))
        # Short exchanges stay together; ReportLab will still split by row if a very
        # large message exceeds the page height.
        story.extend([KeepTogether([turn_table]), Spacer(1, 10)])

    if analysis:
        story.extend([
            PageBreak(),
            Paragraph(_label("POST-SESSION REVIEW", "会后评审", language), eyebrow),
            Paragraph(_label("Conversation Analysis", "对话分析", language), title),
            Paragraph(
                _label(
                    "The following section preserves the generated review as structured headings, findings, and recommendations instead of raw markdown.",
                    "以下内容将生成的评审整理为结构化标题、发现与建议，而不是原始 Markdown。",
                    language,
                ),
                subtitle,
            ),
        ])

        outcome_bg = SUCCESS_BG if str(phase).lower() in {"ended", "closing", "resolved"} else WARNING_BG
        outcome_color = SUCCESS if str(phase).lower() in {"ended", "closing", "resolved"} else WARNING
        review_summary = Table([[
            Paragraph(
                f'<font size="7" color="#6E7F8C">{escape(_label("FINAL STATUS", "最终状态", language))}</font><br/>'
                f'<font size="13" color="{outcome_color.hexval()}"><b>{escape(_phase_display(phase, language))}</b></font>',
                metric_style,
            ),
            Paragraph(
                f'<font size="7" color="#6E7F8C">{escape(_label("EXPERIENCE SCORE", "体验评分", language))}</font><br/>'
                f'<font size="13" color="#173A54"><b>{satisfaction:.0f}/100</b></font>',
                metric_style,
            ),
        ]], colWidths=[doc.width * 0.55, doc.width * 0.45])
        review_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), outcome_bg),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F3F7FA")),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.extend([review_summary, Spacer(1, 14)])

        for kind, text in _analysis_items(analysis):
            if kind == "heading":
                story.append(Paragraph(_inline_markdown(text), analysis_heading))
            elif kind == "bullet":
                story.append(Paragraph(_inline_markdown(text), analysis_bullet, bulletText="•"))
            else:
                story.append(Paragraph(_inline_markdown(text), analysis_body))

    story.extend([
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=4, spaceAfter=7),
        Paragraph(_label("Technical traceability", "技术追溯信息", language), analysis_heading),
        Paragraph(
            f"{escape(_label('Response provider(s)', '回复模型来源', language))}: {_text(_provider_summary(transcript))}<br/>"
            f"{escape(_label('Session ID', '会话 ID', language))}: {_text(session_id)}<br/>"
            f"{escape(_label('Exported', '导出时间', language))}: {_text(generated_at)}",
            footer_style,
        ),
    ])

    def decorate_page(canvas, document) -> None:
        canvas.saveState()
        width, _ = letter
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.45)
        canvas.line(document.leftMargin, 0.48 * inch, width - document.rightMargin, 0.48 * inch)
        canvas.setFont(font, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, 0.30 * inch, f"JSpace Live  |  {session_id}")
        canvas.drawRightString(
            width - document.rightMargin,
            0.30 * inch,
            f"{_label('Page', '第', language)} {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buf.getvalue()
