from __future__ import annotations

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

# Built-in CID font support keeps Simplified Chinese exports self-contained.
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception:
    pass

INK = colors.HexColor("#1F2A33")
MUTED = colors.HexColor("#687781")
NAVY = colors.HexColor("#173A54")
BLUE = colors.HexColor("#2F6F9F")
LINE = colors.HexColor("#DCE5EB")

EMOTION_ZH = {
    "calm": "平静", "neutral": "中性", "curious": "好奇", "hopeful": "有希望",
    "appreciative": "感谢", "satisfied": "满意", "relieved": "安心", "uncertain": "不确定",
    "confused": "困惑", "anxious": "焦虑", "disappointed": "失望", "frustrated": "沮丧",
    "angry": "生气", "impatient": "不耐烦", "skeptical": "怀疑", "distressed": "难受",
    "embarrassed": "尴尬",
}


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
    """Normalize loose markdown into plain report paragraphs."""
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
        if line.startswith(("- ", "• ")):
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


def _phase_display(phase: str, language: str) -> str:
    low = str(phase or "").lower()
    if low == "ended":
        return _label("Ended", "已结束", language)
    if low == "closing":
        return _label("Closing", "收尾中", language)
    if low == "resolved":
        return _label("Resolved", "已解决", language)
    if low == "resolving":
        return _label("Resolving", "解决中", language)
    return _label("Active", "处理中", language)


def _provider_summary(transcript: list[dict]) -> str:
    seen: list[str] = []
    for row in transcript:
        if row.get("role") != "agent":
            continue
        provider = str(row.get("provider") or "").strip().replace("·", "-")
        if provider and provider not in seen:
            seen.append(provider)
    return "; ".join(seen) if seen else "-"


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
    """Export the conversation as a simple, readable text record.

    Conversation messages are deliberately rendered as ordinary flowing text - not
    chat bubbles, cards, or tables. Each utterance is labeled Customer or Support
    Agent, so the PDF reads like a clean transcript and can be archived or copied.
    """
    buf = BytesIO()
    font = _font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.65 * inch,
        title=_label("JSpace Live Conversation", "JSpace Live 对话记录", language),
        author="JSpace Live",
        subject=_label("Conversation transcript and analysis", "对话文本与分析", language),
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "JTitle", parent=styles["Title"], fontName=font, fontSize=18, leading=22,
        textColor=NAVY, spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "JMeta", parent=styles["BodyText"], fontName=font, fontSize=8.5, leading=12,
        textColor=MUTED, spaceAfter=3,
    )
    section_style = ParagraphStyle(
        "JSection", parent=styles["Heading2"], fontName=font, fontSize=13.5, leading=17,
        textColor=NAVY, spaceBefore=10, spaceAfter=8,
    )
    turn_style = ParagraphStyle(
        "JTurn", parent=styles["BodyText"], fontName=font, fontSize=8.2, leading=10,
        textColor=MUTED, spaceBefore=4, spaceAfter=5,
    )
    customer_label_style = ParagraphStyle(
        "JCustomerLabel", parent=styles["BodyText"], fontName=font, fontSize=9.3, leading=11,
        textColor=BLUE, spaceBefore=1, spaceAfter=3,
    )
    agent_label_style = ParagraphStyle(
        "JAgentLabel", parent=customer_label_style, textColor=NAVY,
    )
    message_style = ParagraphStyle(
        "JMessage", parent=styles["BodyText"], fontName=font, fontSize=10.3, leading=15.2,
        textColor=INK, wordWrap="CJK", splitLongWords=True, spaceAfter=5,
    )
    trace_style = ParagraphStyle(
        "JTrace", parent=meta_style, fontSize=7.6, leading=10, spaceAfter=4,
    )
    analysis_heading_style = ParagraphStyle(
        "JAnalysisHeading", parent=styles["Heading3"], fontName=font, fontSize=11.2, leading=14.5,
        textColor=NAVY, spaceBefore=9, spaceAfter=4,
    )
    analysis_body_style = ParagraphStyle(
        "JAnalysisBody", parent=message_style, fontSize=9.8, leading=14.2, spaceAfter=6,
    )
    analysis_bullet_style = ParagraphStyle(
        "JAnalysisBullet", parent=analysis_body_style, leftIndent=13, firstLineIndent=-7,
    )

    patience = float(profile.get("patience", 0) or 0)
    trust = float(profile.get("trust", 0) or 0)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story = [
        Paragraph(_label("JSpace Live Conversation", "JSpace Live 对话记录", language), title_style),
        Paragraph(
            f"{escape(_label('Domain', '领域', language))}: {_text(domain)} | "
            f"{escape(_label('Channel', '渠道', language))}: {_text(channel)}",
            meta_style,
        ),
        Paragraph(
            f"{escape(_label('Session', '会话 ID', language))}: {_text(session_id)} | "
            f"{escape(_label('Status', '状态', language))}: {_text(_phase_display(phase, language))}",
            meta_style,
        ),
        Paragraph(
            f"{escape(_label('Satisfaction', '满意度', language))}: {float(satisfaction):.0f}/100 | "
            f"{escape(_label('Trust', '信任度', language))}: {trust:.0f}/100 | "
            f"{escape(_label('Patience', '耐心度', language))}: {patience:.0f}/100",
            meta_style,
        ),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=9),
        Paragraph(_label("Conversation", "对话", language), section_style),
    ]

    customer_turn = 0
    for row in transcript:
        role = str(row.get("role") or "").strip().lower()
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        if role == "customer":
            customer_turn += 1
            block = [
                Paragraph(f"{escape(_label('Turn', '轮次', language))} {customer_turn:02d}", turn_style),
                Paragraph(_label("Customer", "客户", language), customer_label_style),
                Paragraph(_text(text), message_style),
            ]

            emotion = str(row.get("emotion") or "").strip()
            intensity = row.get("emotion_intensity")
            cue = str(row.get("nonverbal_cue") or "").strip()
            meta_bits: list[str] = []
            if emotion:
                meta_bits.append(_display_emotion(emotion, language))
            if isinstance(intensity, (int, float)):
                meta_bits.append(f"{float(intensity):.0%}")
            if cue:
                meta_bits.append(_display_affect_source(cue, language))
            if meta_bits:
                block.append(Paragraph(
                    f"{escape(_label('Customer affect', '客户情绪', language))}: {_text(' | '.join(meta_bits))}",
                    trace_style,
                ))
            story.append(KeepTogether(block))

        elif role == "agent":
            block = [
                Paragraph(_label("Support Agent", "客服", language), agent_label_style),
                Paragraph(_text(text), message_style),
            ]
            provider = str(row.get("provider") or "").strip().replace("·", "-")
            if provider:
                block.append(Paragraph(
                    f"{escape(_label('Response source', '回复来源', language))}: {_text(provider)}",
                    trace_style,
                ))
            story.append(KeepTogether(block))
            story.extend([
                Spacer(1, 3),
                HRFlowable(width="100%", thickness=0.35, color=LINE, spaceBefore=2, spaceAfter=7),
            ])
        else:
            # Preserve unexpected transcript rows rather than silently dropping data.
            story.append(Paragraph(_label("Conversation note", "对话备注", language), customer_label_style))
            story.append(Paragraph(_text(text), message_style))

    if analysis:
        story.extend([
            PageBreak(),
            Paragraph(_label("Conversation Analysis", "对话分析", language), section_style),
        ])
        for kind, text in _analysis_items(analysis):
            if kind == "heading":
                story.append(Paragraph(_inline_markdown(text), analysis_heading_style))
            elif kind == "bullet":
                story.append(Paragraph(_inline_markdown(text), analysis_bullet_style, bulletText="-"))
            else:
                story.append(Paragraph(_inline_markdown(text), analysis_body_style))

    story.extend([
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=3, spaceAfter=6),
        Paragraph(
            f"{escape(_label('Response provider(s)', '回复模型来源', language))}: {_text(_provider_summary(transcript))}<br/>"
            f"{escape(_label('Exported', '导出时间', language))}: {_text(generated_at)}",
            trace_style,
        ),
    ])

    def decorate_page(canvas, document) -> None:
        canvas.saveState()
        width, _ = letter
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.35)
        canvas.line(document.leftMargin, 0.45 * inch, width - document.rightMargin, 0.45 * inch)
        canvas.setFont(font, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, 0.28 * inch, f"JSpace Live | {session_id}")
        canvas.drawRightString(
            width - document.rightMargin,
            0.28 * inch,
            f"{_label('Page', '第', language)} {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buf.getvalue()


def build_plain_transcript_pdf(**kwargs) -> bytes:
    """Descriptive alias for the canonical plain-text conversation exporter.

    The website intentionally imports ``build_conversation_pdf`` because that
    name existed in earlier deployments. Keeping the stable import contract
    prevents Streamlit Cloud hot-reloads from failing when ``app.py`` updates
    before a cached ``conversation_export`` module is refreshed.
    """
    return build_conversation_pdf(**kwargs)
