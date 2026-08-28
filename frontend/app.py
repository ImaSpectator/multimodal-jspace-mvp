from __future__ import annotations

import html
import inspect
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.jspace.ai_provider import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_VIDEO_MODEL,
    analyze_media_for_jspace,
    analyze_conversation_summary,
    enhance_scenario_with_deepseek,
    generate_support_reply,
    probe_deepseek,
    stream_support_reply,
)
from backend.jspace.engine import merge_concepts, refresh_state  # noqa: E402
from backend.jspace.scenario_generator import generate_manual_context, generate_scenario, list_domains  # noqa: E402
from backend.jspace.schemas import ImageObservation, ScenarioControls  # noqa: E402
from backend.jspace.localization import conflict_description_zh  # noqa: E402
from backend.jspace.simulator import (  # noqa: E402
    append_agent_reply,
    apply_manual_customer_message,
    apply_scenario_customer_step,
    end_manual_session,
    end_scenario_session,
    new_manual_state,
    new_scenario_state,
)

try:  # v1.1 backend; tolerate a stale Streamlit module during redeploy.
    from backend.jspace.simulator import update_customer_relationship as _backend_update_customer_relationship  # noqa: E402
except ImportError:  # pragma: no cover - compatibility path for stale deployments
    _backend_update_customer_relationship = None



# Canonical website PDF renderer.
# IMPORTANT: this implementation lives in app.py intentionally so the Streamlit
# download button cannot fall back to an older cached backend PDF renderer during
# a hot deploy. It is the same plain-transcript layout used for the approved
# v1.4.3 English and Chinese sample PDFs.
# Built-in CID font support keeps Simplified Chinese exports self-contained.
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception:
    pass

_PDF_INK = colors.HexColor("#1F2A33")
_PDF_MUTED = colors.HexColor("#687781")
_PDF_NAVY = colors.HexColor("#173A54")
_PDF_BLUE = colors.HexColor("#2F6F9F")
_PDF_LINE = colors.HexColor("#DCE5EB")

_PDF_EMOTION_ZH = {
    "calm": "平静", "neutral": "中性", "curious": "好奇", "hopeful": "有希望",
    "appreciative": "感谢", "satisfied": "满意", "relieved": "安心", "uncertain": "不确定",
    "confused": "困惑", "anxious": "焦虑", "disappointed": "失望", "frustrated": "沮丧",
    "angry": "生气", "impatient": "不耐烦", "skeptical": "怀疑", "distressed": "难受",
    "embarrassed": "尴尬",
}


def _pdf_font(language: str) -> str:
    return "STSong-Light" if str(language).lower().startswith(("zh", "chinese", "simplified")) else "Helvetica"


def _pdf_label(en: str, zh: str, language: str) -> str:
    return zh if _pdf_font(language) == "STSong-Light" else en


def _pdf_text(value: object) -> str:
    return escape(str(value or "")).replace("\n", "<br/>")


def _pdf_inline_markdown(value: str) -> str:
    safe = escape(str(value or ""))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe)


def _pdf_analysis_items(analysis: str) -> list[tuple[str, str]]:
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


def _pdf_display_emotion(emotion: str, language: str) -> str:
    key = str(emotion or "").strip().lower()
    if not key:
        return ""
    if _pdf_font(language) == "STSong-Light":
        return _PDF_EMOTION_ZH.get(key, key)
    return key.replace("_", " ").title()


def _pdf_display_affect_source(value: str, language: str) -> str:
    text = str(value or "").strip()
    if _pdf_font(language) != "STSong-Light":
        return text
    low = text.lower()
    if "text-derived" in low or "inferred from text" in low:
        return "文字推断情绪"
    if "audio-derived" in low or "inferred from audio" in low:
        return "音频转写文字推断情绪"
    if "video-derived" in low or "inferred from video" in low:
        return "视频内容推断情绪"
    return text


def _pdf_phase_display(phase: str, language: str) -> str:
    low = str(phase or "").lower()
    if low == "ended":
        return _pdf_label("Ended", "已结束", language)
    if low == "closing":
        return _pdf_label("Closing", "收尾中", language)
    if low == "resolved":
        return _pdf_label("Resolved", "已解决", language)
    if low == "resolving":
        return _pdf_label("Resolving", "解决中", language)
    return _pdf_label("Active", "处理中", language)


def _pdf_provider_summary(transcript: list[dict]) -> str:
    seen: list[str] = []
    for row in transcript:
        if row.get("role") != "agent":
            continue
        provider = str(row.get("provider") or "").strip().replace("·", "-")
        if provider and provider not in seen:
            seen.append(provider)
    return "; ".join(seen) if seen else "-"


def build_website_plain_transcript_pdf(
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
    font = _pdf_font(language)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.65 * inch,
        title=_pdf_label("JSpace Live Conversation", "JSpace Live 对话记录", language),
        author="JSpace Live",
        subject=_pdf_label("Conversation transcript and analysis", "对话文本与分析", language),
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "JTitle", parent=styles["Title"], fontName=font, fontSize=18, leading=22,
        textColor=_PDF_NAVY, spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "JMeta", parent=styles["BodyText"], fontName=font, fontSize=8.5, leading=12,
        textColor=_PDF_MUTED, spaceAfter=3,
    )
    section_style = ParagraphStyle(
        "JSection", parent=styles["Heading2"], fontName=font, fontSize=13.5, leading=17,
        textColor=_PDF_NAVY, spaceBefore=10, spaceAfter=8,
    )
    turn_style = ParagraphStyle(
        "JTurn", parent=styles["BodyText"], fontName=font, fontSize=8.2, leading=10,
        textColor=_PDF_MUTED, spaceBefore=4, spaceAfter=5,
    )
    customer_label_style = ParagraphStyle(
        "JCustomerLabel", parent=styles["BodyText"], fontName=font, fontSize=9.3, leading=11,
        textColor=_PDF_BLUE, spaceBefore=1, spaceAfter=3,
    )
    agent_label_style = ParagraphStyle(
        "JAgentLabel", parent=customer_label_style, textColor=_PDF_NAVY,
    )
    message_style = ParagraphStyle(
        "JMessage", parent=styles["BodyText"], fontName=font, fontSize=10.3, leading=15.2,
        textColor=_PDF_INK, wordWrap="CJK", splitLongWords=True, spaceAfter=5,
    )
    trace_style = ParagraphStyle(
        "JTrace", parent=meta_style, fontSize=7.6, leading=10, spaceAfter=4,
    )
    analysis_heading_style = ParagraphStyle(
        "JAnalysisHeading", parent=styles["Heading3"], fontName=font, fontSize=11.2, leading=14.5,
        textColor=_PDF_NAVY, spaceBefore=9, spaceAfter=4,
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
        Paragraph(_pdf_label("JSpace Live Conversation", "JSpace Live 对话记录", language), title_style),
        Paragraph(
            f"{escape(_pdf_label('Domain', '领域', language))}: {_pdf_text(domain)} | "
            f"{escape(_pdf_label('Channel', '渠道', language))}: {_pdf_text(channel)}",
            meta_style,
        ),
        Paragraph(
            f"{escape(_pdf_label('Session', '会话 ID', language))}: {_pdf_text(session_id)} | "
            f"{escape(_pdf_label('Status', '状态', language))}: {_pdf_text(_pdf_phase_display(phase, language))}",
            meta_style,
        ),
        Paragraph(
            f"{escape(_pdf_label('Satisfaction', '满意度', language))}: {float(satisfaction):.0f}/100 | "
            f"{escape(_pdf_label('Trust', '信任度', language))}: {trust:.0f}/100 | "
            f"{escape(_pdf_label('Patience', '耐心度', language))}: {patience:.0f}/100",
            meta_style,
        ),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=0.6, color=_PDF_LINE, spaceAfter=9),
        Paragraph(_pdf_label("Conversation", "对话", language), section_style),
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
                Paragraph(f"{escape(_pdf_label('Turn', '轮次', language))} {customer_turn:02d}", turn_style),
                Paragraph(_pdf_label("Customer", "客户", language), customer_label_style),
                Paragraph(_pdf_text(text), message_style),
            ]

            emotion = str(row.get("emotion") or "").strip()
            intensity = row.get("emotion_intensity")
            cue = str(row.get("nonverbal_cue") or "").strip()
            meta_bits: list[str] = []
            if emotion:
                meta_bits.append(_pdf_display_emotion(emotion, language))
            if isinstance(intensity, (int, float)):
                meta_bits.append(f"{float(intensity):.0%}")
            if cue:
                meta_bits.append(_pdf_display_affect_source(cue, language))
            if meta_bits:
                block.append(Paragraph(
                    f"{escape(_pdf_label('Customer affect', '客户情绪', language))}: {_pdf_text(' | '.join(meta_bits))}",
                    trace_style,
                ))
            story.append(KeepTogether(block))

        elif role == "agent":
            block = [
                Paragraph(_pdf_label("Support Agent", "客服", language), agent_label_style),
                Paragraph(_pdf_text(text), message_style),
            ]
            provider = str(row.get("provider") or "").strip().replace("·", "-")
            if provider:
                block.append(Paragraph(
                    f"{escape(_pdf_label('Response source', '回复来源', language))}: {_pdf_text(provider)}",
                    trace_style,
                ))
            story.append(KeepTogether(block))
            story.extend([
                Spacer(1, 3),
                HRFlowable(width="100%", thickness=0.35, color=_PDF_LINE, spaceBefore=2, spaceAfter=7),
            ])
        else:
            # Preserve unexpected transcript rows rather than silently dropping data.
            story.append(Paragraph(_pdf_label("Conversation note", "对话备注", language), customer_label_style))
            story.append(Paragraph(_pdf_text(text), message_style))

    if analysis:
        story.extend([
            PageBreak(),
            Paragraph(_pdf_label("Conversation Analysis", "对话分析", language), section_style),
        ])
        for kind, text in _pdf_analysis_items(analysis):
            if kind == "heading":
                story.append(Paragraph(_pdf_inline_markdown(text), analysis_heading_style))
            elif kind == "bullet":
                story.append(Paragraph(_pdf_inline_markdown(text), analysis_bullet_style, bulletText="-"))
            else:
                story.append(Paragraph(_pdf_inline_markdown(text), analysis_body_style))

    story.extend([
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=_PDF_LINE, spaceBefore=3, spaceAfter=6),
        Paragraph(
            f"{escape(_pdf_label('Response provider(s)', '回复模型来源', language))}: {_pdf_text(_pdf_provider_summary(transcript))}<br/>"
            f"{escape(_pdf_label('Exported', '导出时间', language))}: {_pdf_text(generated_at)}",
            trace_style,
        ),
    ])

    def decorate_page(canvas, document) -> None:
        canvas.saveState()
        width, _ = letter
        canvas.setStrokeColor(_PDF_LINE)
        canvas.setLineWidth(0.35)
        canvas.line(document.leftMargin, 0.45 * inch, width - document.rightMargin, 0.45 * inch)
        canvas.setFont(font, 7)
        canvas.setFillColor(_PDF_MUTED)
        canvas.drawString(document.leftMargin, 0.28 * inch, f"JSpace Live | {session_id}")
        canvas.drawRightString(
            width - document.rightMargin,
            0.28 * inch,
            f"{_pdf_label('Page', '第', language)} {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buf.getvalue()



def update_customer_relationship(profile: dict, state, reply: str, provider: str = "") -> None:
    """Update Patience/Trust even when an older backend is temporarily loaded."""
    if _backend_update_customer_relationship is not None:
        _backend_update_customer_relationship(profile, state, reply, provider)
        return
    if not profile:
        return
    low = (reply or "").lower()
    provider_low = (provider or "").lower()
    agent_turns = sum(1 for row in getattr(state, "transcript", []) if row.get("role") == "agent")
    concrete_progress = any(token in low for token in [
        "verified", "confirmed", "found", "identified", "root cause", "fixed", "resolved", "updated",
        "removed", "unlocked", "reissued", "refunded", "activated", "next step", "i'll apply", "i will apply",
        "已经核实", "已经确认", "查到", "根因", "已解决", "已修复", "已更新", "下一步", "我会处理", "已经处理",
    ])
    asks_repeat = any(token in low for token in [
        "try again", "restart again", "reset again", "repeat the", "do that again",
        "再试一次", "再重启", "再重置", "重复刚才",
    ])
    fallback = "fallback" in provider_low or "simulation" in provider_low
    prolonged_conflict = bool(getattr(state, "conflicts", [])) and agent_turns >= 3 and getattr(state, "session_phase", "active") not in {"resolved", "closing", "ended"}
    patience_loss = (6.0 if fallback else 0.0) + (7.0 if asks_repeat else 0.0)
    if prolonged_conflict and not concrete_progress:
        patience_loss += 3.0 + min(3.0, max(0, agent_turns - 3) * 0.7)
    if not concrete_progress and agent_turns >= 4 and getattr(state, "session_phase", "active") == "active":
        patience_loss += 1.5
    profile["patience"] = int(round(min(100.0, float(profile.get("patience", 85)) - patience_loss)))
    trust_delta = 0.0
    if getattr(state, "session_phase", "active") in {"resolved", "closing"} or any(x in low for x in ["confirmed resolved", "issue is resolved", "已经解决", "确认已经解决"]):
        trust_delta += 4.0
    elif concrete_progress:
        trust_delta += 1.8
    if fallback:
        trust_delta -= 3.5
    if asks_repeat:
        trust_delta -= 2.5
    if prolonged_conflict and not concrete_progress:
        trust_delta -= 1.5
    if abs(trust_delta) > 0.01:
        trust_delta += ((sum(ord(ch) for ch in (reply or "")) + agent_turns) % 3 - 1) * 0.6
    profile["trust"] = int(round(max(0.0, min(100.0, float(profile.get("trust", 55)) + trust_delta))))


APP_VERSION = "1.4.3-plain-transcript-bilingual-conflicts"

DOMAIN_DESCRIPTIONS = {
    "account_access": "Login, authentication, identity verification, lockouts, and account recovery.",
    "banking_fraud": "Suspicious transactions, card security, disputes, replacement, and fraud remediation.",
    "delivery": "Missing, delayed, incorrectly scanned, or misdelivered packages and shipments.",
    "device_support": "Connected devices, firmware, pairing, hardware issues, and smart-device troubleshooting.",
    "event_ticketing": "Ticket transfers, barcode activation, access rights, and event-entry problems.",
    "healthcare_appointment": "Scheduling, appointment confirmation, provider availability, and booking mismatches.",
    "hotel_hospitality": "Reservations, property-system mismatches, room availability, and booking changes.",
    "insurance_claim": "Claim status, missing documentation, review blockers, and next-step requirements.",
    "internet": "Home internet outages, modem/router state, network incidents, and connectivity diagnostics.",
    "marketplace_dispute": "Buyer/seller disputes, replacements, fulfillment remedies, and case resolution.",
    "payment": "Declines, failed authorizations, duplicate-charge risk, and checkout/payment troubleshooting.",
    "return_refund": "Returns, refund status, warehouse processing, and reimbursement delays.",
    "rideshare": "Trip charges, authorization holds, driver/rider disputes, and account adjustments.",
    "software_saas": "Permissions, entitlements, access, workspace configuration, and SaaS service problems.",
    "subscription": "Cancellation, renewal, billing status, and recurring-subscription issues.",
    "telecom_mobile": "Mobile plans, provisioning, data limits, network policy, and carrier-account issues.",
    "travel": "Flight changes, ticketing, reissues, booking status, and itinerary support.",
    "utilities": "Metering, utility bills, service records, and account/usage discrepancies.",
}

CHANNELS = {
    "Text Messages": {
        "icon": "💬", "slug": "text messages",
        "hint": "Text-first support. Customer wording is the primary signal.",
        "affect_source": "text",
    },
    "Image Upload": {
        "icon": "🖼️", "slug": "image evidence",
        "hint": "Image-led support. Upload screenshots or photos; the chat can reference only visual evidence for this turn.",
        "affect_source": "image",
    },
    "Audio Upload": {
        "icon": "🎧", "slug": "audio evidence",
        "hint": "Audio-led support. Uploaded audio is transcribed by Hy-ASR; emotion is inferred from the transcript, not raw vocal tone.",
        "affect_source": "audio",
    },
    "Video Upload": {
        "icon": "◉", "slug": "video evidence",
        "hint": "Video-led support. YT-VITA analyzes uploaded video frames and its audio track for this turn.",
        "affect_source": "video",
    },
    "Voice Call": {
        "icon": "📞", "slug": "voice call",
        "hint": "Live-call style support. Customer wording is shown like a call transcript, with audio-style affect cues.",
        "affect_source": "audio",
    },
    "Video + Voice": {
        "icon": "📹", "slug": "video + voice call",
        "hint": "Live video-call support. Video evidence and spoken context are interpreted together.",
        "affect_source": "video",
    },
    "Multimodal Mix": {
        "icon": "✦", "slug": "multimodal conversation",
        "hint": "DeepSeek image/text + Hy-ASR audio transcription + YT-VITA video + backend evidence in one shared workspace.",
        "affect_source": "audio",
    },
}
SCENARIO_CHANNELS = ["Text Messages", "Voice Call", "Video + Voice", "Multimodal Mix"]
MANUAL_MODE_CONFIG = {
    "Text Messages": {"allow_text": True, "show_suggestion": True, "file_types": [], "allow_video_url": False, "placeholder": ("Type the customer's text message…", "输入客户的文字消息…")},
    "Image Upload": {"allow_text": False, "show_suggestion": False, "file_types": ["png", "jpg", "jpeg", "webp"], "allow_video_url": False, "placeholder": ("Image-only mode", "仅图片模式")},
    "Audio Upload": {"allow_text": False, "show_suggestion": False, "file_types": ["mp3", "wav", "m4a", "ogg"], "allow_video_url": False, "placeholder": ("Audio-only mode", "仅音频模式")},
    "Video Upload": {"allow_text": False, "show_suggestion": False, "file_types": ["mp4", "mov", "avi", "webm"], "allow_video_url": True, "placeholder": ("Video-only mode", "仅视频模式")},
    "Multimodal Mix": {"allow_text": True, "show_suggestion": True, "file_types": ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "mp4", "mov", "avi", "webm"], "allow_video_url": True, "placeholder": ("Type the customer's message, then add any supporting media…", "先输入客户消息，再添加支持性媒体…")},
}

CUSTOMER_STARTERS = {
    "account_access": "I can't get back into my account even though I completed the verification steps. Can you check what's still blocking me?",
    "banking_fraud": "I don't recognize a transaction on my account and I need to know whether my card is secure.",
    "delivery": "My tracking information doesn't match what I'm actually seeing. Can you check where the package really is?",
    "device_support": "My device keeps disconnecting even after I retried the setup. Can you help me figure out what's actually wrong?",
    "event_ticketing": "My ticket looks available in the app, but I'm worried it won't scan at the event. Can you verify it?",
    "healthcare_appointment": "My appointment looks confirmed in one place but not another. Can you verify what is actually booked?",
    "hotel_hospitality": "My reservation details don't match what I was promised. Can you confirm what the hotel actually has on file?",
    "insurance_claim": "My claim looks like it's progressing, but I still haven't received a clear answer. What's holding it up?",
    "internet": "My internet keeps dropping even though I've already restarted the equipment. Can you check the network side?",
    "marketplace_dispute": "The seller says the issue is resolved, but I haven't received the promised remedy. Can you check the case?",
    "payment": "My payment keeps failing and I've already retried it. Can you check why instead of asking me to try again?",
    "return_refund": "My return shows completed, but I still don't have the refund. Can you check the actual refund status?",
    "rideshare": "I still see a charge from a ride issue and I want to understand whether it's a real charge or a hold.",
    "software_saas": "I can open the workspace, but I still don't have the access I need. Can you check my entitlement?",
    "subscription": "I cancelled my subscription but I'm not sure the billing system actually stopped renewal. Can you verify it?",
    "telecom_mobile": "My plan says it changed, but my phone is still behaving like the old plan is active. Can you check provisioning?",
    "travel": "My itinerary looks updated in the app, but I want to confirm the ticket is actually reissued before I travel.",
    "utilities": "My bill doesn't match what I expected from the meter reading. Can you check which record is authoritative?",
}

CUSTOMER_STARTERS_ZH = {
    "account_access": "我已经完成验证步骤，但还是无法登录账户。你能帮我查一下还有什么在阻止我吗？",
    "banking_fraud": "我的账户里有一笔我不认识的交易，我需要确认银行卡现在是否安全。",
    "delivery": "物流追踪信息和我实际看到的情况不一致。你能帮我确认包裹现在到底在哪里吗？",
    "device_support": "设备一直断开连接，我已经重新设置过了。你能帮我查清楚真正的问题吗？",
    "event_ticketing": "票在应用里看起来可以使用，但我担心现场无法扫码。你能帮我确认吗？",
    "healthcare_appointment": "我的预约在一个地方显示已确认，但另一个地方又不一致。你能确认实际预约状态吗？",
    "hotel_hospitality": "我的酒店预订信息和之前承诺的不一致。你能确认酒店系统里实际记录的内容吗？",
    "insurance_claim": "理赔看起来在推进，但我一直没有收到明确答复。现在到底卡在哪里？",
    "internet": "网络一直掉线，我已经重启过设备了。你能帮我检查一下网络侧的问题吗？",
    "marketplace_dispute": "卖家说问题已经解决，但我还没有收到承诺的补救。你能帮我查一下案件状态吗？",
    "payment": "我的支付一直失败，而且我已经重试过了。你能查一下原因，而不是再让我重试吗？",
    "return_refund": "退货显示已经完成，但我还是没有收到退款。你能查一下实际退款状态吗？",
    "rideshare": "我在一次行程问题后仍然看到一笔扣款，我想确认这是真实扣款还是预授权。",
    "software_saas": "我可以打开工作区，但仍然没有需要的权限。你能帮我检查授权状态吗？",
    "subscription": "我已经取消订阅，但不确定系统是否真的停止续费。你能确认吗？",
    "telecom_mobile": "我的套餐显示已经变更，但手机表现还是像旧套餐。你能检查一下开通状态吗？",
    "travel": "应用里的行程已经更新，但我想确认机票是否真的完成重签。",
    "utilities": "账单金额和电表读数不一致。你能帮我确认哪个记录才是权威的吗？",
}

st.set_page_config(
    page_title="JSpace Live — Multimodal Customer Service",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --j-cyan:#5DF5FF; --j-blue:#5C8DFF; --j-violet:#B174FF; --j-pink:#FF72C7;
  --j-bg:#050812; --j-panel:rgba(12,22,40,.72); --j-border:rgba(104,226,255,.18);
  --j-text:#EDF7FF; --j-muted:#8EA6C0; --j-green:#7DFFBD; --j-red:#FF7D9D;
}
.stApp {
  background:
    radial-gradient(circle at 8% 4%, rgba(75,126,255,.20), transparent 30%),
    radial-gradient(circle at 94% 14%, rgba(177,116,255,.15), transparent 27%),
    radial-gradient(circle at 55% 88%, rgba(35,224,226,.07), transparent 31%),
    linear-gradient(180deg,#060A14 0%,#040710 58%,#060A12 100%);
  color:var(--j-text);
}
.block-container { max-width:1500px; padding-top:1.05rem; padding-bottom:4rem; }
header, [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none!important; }
a.anchor-link, [data-testid="stMarkdownContainer"] h1 > a, [data-testid="stMarkdownContainer"] h2 > a, [data-testid="stMarkdownContainer"] h3 > a, [data-testid="stMarkdownContainer"] h4 > a { display:none!important; }
.j-hero { padding:1.45rem 1.65rem; border:1px solid var(--j-border); border-radius:22px; background:linear-gradient(135deg,rgba(16,31,59,.90),rgba(8,15,31,.76)); box-shadow:0 20px 70px rgba(0,0,0,.31), inset 0 1px 0 rgba(255,255,255,.04); position:relative; overflow:hidden; margin-bottom:.7rem; }
.j-hero:before { content:""; position:absolute; top:0; left:-25%; width:55%; height:2px; background:linear-gradient(90deg,transparent,var(--j-cyan),var(--j-violet),transparent); animation:scan 5s linear infinite; }
@keyframes scan { from{transform:translateX(0)} to{transform:translateX(230%)} }
.j-kicker { color:var(--j-cyan); letter-spacing:.20em; font-size:.70rem; font-weight:800; }
.j-title { font-size:2.25rem; line-height:1.08; font-weight:760; margin:.34rem 0 .42rem; color:#F8FBFF; }
.j-sub { color:var(--j-muted); max-width:1050px; font-size:.98rem; line-height:1.55; }
.j-pill { display:inline-block; padding:.20rem .53rem; border-radius:999px; border:1px solid rgba(97,244,255,.25); background:rgba(97,244,255,.07); color:#C5FBFF; font-size:.72rem; margin:.6rem .35rem 0 0; }
.j-card { border:1px solid var(--j-border); border-radius:16px; background:var(--j-panel); padding:.92rem 1rem; margin:.42rem 0; box-shadow:inset 0 1px 0 rgba(255,255,255,.025); }
.j-card-title { color:#B8DDF5; font-size:.68rem; text-transform:uppercase; letter-spacing:.10em; margin-bottom:.25rem; }
.j-card-value { color:#F2F8FF; font-size:1rem; font-weight:650; overflow-wrap:anywhere; line-height:1.4; }
.j-card-meta { color:var(--j-muted); font-size:.75rem; margin-top:.28rem; }
.j-next { border-color:rgba(93,245,255,.35); background:linear-gradient(135deg,rgba(11,72,84,.34),rgba(28,36,86,.46)); box-shadow:0 0 28px rgba(70,210,255,.06); }
.j-case { border-color:rgba(177,116,255,.30); background:linear-gradient(135deg,rgba(52,25,83,.30),rgba(16,26,55,.55)); }
.j-conflict { border-color:rgba(255,185,91,.38); background:rgba(91,51,17,.25); }
.j-concept { border-left:3px solid var(--j-blue); }
.j-concept.disputed { border-left-color:#FFB45F; } .j-concept.unresolved { border-left-color:#FF6D91; }
.j-profile-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.54rem; margin:.35rem 0 .75rem; }
.j-profile-cell { border:1px solid rgba(126,168,214,.14); background:rgba(10,19,35,.58); border-radius:12px; padding:.60rem .72rem; min-width:0; }
.j-profile-label { color:#7891AD; font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; }
.j-profile-value { color:#ECF5FF; font-size:.88rem; font-weight:620; margin-top:.14rem; overflow-wrap:anywhere; }
.j-emotion { border:1px solid rgba(177,116,255,.25); background:rgba(63,38,90,.18); border-radius:14px; padding:.68rem .82rem; min-height:88px; }
.j-emotion-label { color:#9E86C0; font-size:.64rem; text-transform:uppercase; letter-spacing:.09em; }
.j-emotion-value { color:#F4E9FF; font-weight:750; line-height:1.08; margin-top:.24rem; overflow-wrap:anywhere; word-break:break-word; }
.j-phone { border:1px solid rgba(108,195,255,.19); border-radius:24px; background:linear-gradient(180deg,rgba(6,12,24,.88),rgba(7,13,25,.72)); padding:.75rem .78rem 1rem; box-shadow:0 18px 55px rgba(0,0,0,.20), inset 0 0 38px rgba(51,117,216,.035); height:min(58vh,590px); min-height:420px; max-height:590px; overflow-y:scroll; overscroll-behavior:contain; scroll-behavior:smooth; scrollbar-gutter:stable; }
.j-phone::-webkit-scrollbar { width:8px; }
.j-phone::-webkit-scrollbar-track { background:rgba(8,17,31,.28); border-radius:10px; }
.j-phone::-webkit-scrollbar-thumb { background:rgba(93,245,255,.24); border-radius:10px; }
.j-phone::-webkit-scrollbar-thumb:hover { background:rgba(93,245,255,.42); }
.j-phone-head { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(109,173,220,.12); padding:.38rem .32rem .70rem; margin-bottom:.62rem; position:sticky; top:0; background:rgba(6,12,24,.94); z-index:2; }
.j-channel-name { color:#EAF7FF; font-size:.90rem; font-weight:700; } .j-channel-meta { color:#6F8AA8; font-size:.69rem; }
.j-live-dot { display:inline-block; width:7px; height:7px; background:#6DFFB3; border-radius:50%; box-shadow:0 0 10px rgba(109,255,179,.8); margin-right:.35rem; animation:pulse 1.8s infinite; }
@keyframes pulse { 50%{opacity:.45; transform:scale(.85)} }
.j-msg-row { display:flex; margin:.52rem .18rem; }
.j-msg-row.customer { justify-content:flex-end; } .j-msg-row.agent { justify-content:flex-start; }
.j-msg { max-width:80%; border-radius:17px; padding:.66rem .82rem; line-height:1.45; font-size:.91rem; border:1px solid rgba(128,178,223,.12); box-shadow:0 8px 20px rgba(0,0,0,.10); }
.j-msg.customer { background:linear-gradient(135deg,rgba(43,95,167,.78),rgba(56,71,149,.74)); color:#F5FAFF; border-bottom-right-radius:5px; }
.j-msg.agent { background:rgba(17,28,47,.88); color:#EDF6FF; border-bottom-left-radius:5px; }
.j-msg-meta { color:#8BA6C2; font-size:.65rem; margin-top:.35rem; line-height:1.35; }
.j-typing { color:#6CEAFF; font-size:.85rem; animation:pulse 1.2s infinite; }
.j-node-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.55rem; margin:.6rem 0 1rem; }
.j-node { border:1px solid rgba(95,207,255,.17); background:linear-gradient(150deg,rgba(17,33,57,.68),rgba(10,18,34,.62)); border-radius:15px; padding:.8rem; min-height:118px; }
.j-node-num { color:var(--j-cyan); font-size:.65rem; letter-spacing:.12em; } .j-node-name { font-weight:720; color:#EEF7FF; margin:.25rem 0; } .j-node-desc { color:#8EA6C0; font-size:.76rem; line-height:1.4; }
.j-domain { border:1px solid rgba(126,168,214,.13); border-radius:12px; padding:.70rem .78rem; background:rgba(10,19,34,.49); min-height:98px; margin:.3rem 0; }
.j-domain strong { color:#DDF3FF; } .j-domain span { color:#829AB5; font-size:.76rem; line-height:1.35; display:block; margin-top:.22rem; }
.j-suggest { border:1px dashed rgba(93,245,255,.25); background:rgba(22,51,75,.22); border-radius:14px; padding:.72rem .85rem; color:#AFC8DB; font-style:italic; margin:.55rem 0; }
.j-utility { color:#A8C6DD; font-size:.75rem; padding:.35rem 0; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,var(--j-blue),var(--j-violet),var(--j-pink)); }
.stTabs [data-baseweb="tab-list"] { gap:.42rem; } .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.46rem .88rem; background:rgba(11,21,39,.62); }
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stPopover"] button { border-radius:12px!important; border:1px solid rgba(93,245,255,.23)!important; background:rgba(11,21,39,.72); color:#EDF7FF; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background:linear-gradient(100deg,#3179D9,#6254D8,#7A4EC5)!important; border:1px solid rgba(116,218,255,.42)!important; color:white!important; box-shadow:0 9px 26px rgba(69,97,213,.20); }
[data-testid="stFileUploaderDropzone"] { background:rgba(8,17,32,.58); border-color:rgba(93,245,255,.18); }
hr { border-color:rgba(140,175,215,.12)!important; }
@media(max-width:1000px){ .j-profile-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.j-node-grid{grid-template-columns:1fr 1fr}.j-title{font-size:1.72rem}.j-msg{max-width:94%} }

/* v1.3.1 stable header toolbar. Keep it in normal Streamlit flow — never fixed-position
   the column container, because Streamlit grid widths collapse when detached from layout. */
.st-key-utility_toolbar {
  margin-top:-1.05rem!important; margin-bottom:.05rem!important; padding:0!important; background:transparent!important;
}
.st-key-utility_toolbar [data-testid="stHorizontalBlock"] {
  align-items:center!important; gap:.10rem!important; width:100%!important;
}
.st-key-utility_toolbar [data-testid="column"] { padding:0!important; margin:0!important; }
.st-key-utility_toolbar .stButton { margin:0!important; padding:0!important; width:100%!important; }
.st-key-utility_toolbar .stButton > button {
  width:100%!important; height:2.15rem!important; min-height:2.15rem!important; padding:0!important; margin:0!important;
  border:0!important; outline:0!important; background:transparent!important; box-shadow:none!important;
  color:#DCE9F6!important; display:flex!important; align-items:center!important; justify-content:center!important;
}
.st-key-utility_toolbar .stButton > button:hover { background:rgba(118,165,210,.07)!important; border-radius:8px!important; color:#FFFFFF!important; }
.st-key-utility_toolbar .stButton > button:focus,
.st-key-utility_toolbar .stButton > button:focus-visible { outline:none!important; box-shadow:none!important; }
.st-key-utility_toolbar .stButton > button > div {
  width:100%!important; height:100%!important; display:flex!important; align-items:center!important; justify-content:center!important;
  gap:0!important; margin:0!important; padding:0!important;
}
.st-key-top_help .stButton > button p,
.st-key-top_share .stButton > button p,
.st-key-top_reset .stButton > button p,
.st-key-top_settings .stButton > button p { display:none!important; }
.st-key-utility_toolbar .stButton > button [data-testid="stIconMaterial"] {
  display:block!important; width:1.18rem!important; height:1.18rem!important; min-width:1.18rem!important;
  margin:0!important; padding:0!important; font-size:1.18rem!important; line-height:1.18rem!important; text-align:center!important;
}
/* Material glyph optical centers sit slightly left inside Streamlit's icon wrapper.
   Keep the four right-side utility icons on the same 4px optical offset. */
.st-key-top_help [data-testid="stIconMaterial"],
.st-key-top_share [data-testid="stIconMaterial"],
.st-key-top_reset [data-testid="stIconMaterial"] {
  transform:translateX(4px)!important;
}
.st-key-top_settings [data-testid="stIconMaterial"] {
  transform:translateX(4px)!important;
}
.st-key-top_language .stButton > button p {
  display:block!important; width:100%!important; margin:0!important; padding:0!important; font-size:.74rem!important;
  font-weight:760!important; line-height:2.15rem!important; text-align:center!important; white-space:nowrap!important;
}
@media(max-width:700px){ .st-key-utility_toolbar{margin-top:-.65rem!important;} }
/* Manual composer stays visually attached to the conversation. The actual text entry
   is a Streamlit form so Enter and Send behave identically; the suggestion lives beside it. */
.st-key-manual_composer { margin-top:.42rem; margin-bottom:.28rem; padding:.72rem .76rem; border:1px solid rgba(93,245,255,.16); border-radius:16px; background:rgba(8,17,31,.74); box-shadow:inset 0 1px 0 rgba(255,255,255,.025); }
.st-key-manual_composer [data-testid="stHorizontalBlock"] { align-items:end!important; gap:.52rem!important; }
.st-key-manual_composer [data-testid="stForm"] { border:0!important; padding:0!important; background:transparent!important; }
.st-key-manual_composer [data-testid="stTextInput"] { margin:0!important; }
.st-key-manual_composer [data-testid="stTextInput"] input { min-height:2.72rem; border-radius:12px!important; background:rgba(5,12,24,.84)!important; border:1px solid rgba(115,184,232,.22)!important; }
.st-key-manual_composer [data-testid="stFormSubmitButton"] button { min-height:2.72rem!important; margin:0!important; }
.st-key-manual_composer .stButton>button { min-height:2.40rem!important; }
.j-suggest-mini { border:1px solid rgba(93,245,255,.18); background:linear-gradient(145deg,rgba(22,51,75,.24),rgba(24,30,64,.22)); border-radius:12px; padding:.52rem .62rem; color:#B8D0E2; font-size:.72rem; line-height:1.34; margin-bottom:.34rem; max-height:4.85rem; overflow-y:auto; }
#scenario-live-anchor { scroll-margin-top: 20px; }

</style>
""",
    unsafe_allow_html=True,
)


def _secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name, None)
        if value is not None:
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return str(value) if value is not None else default


TOKENHUB_API_KEY = _secret("TOKENHUB_API_KEY")
TOKENHUB_MODEL = _secret("TOKENHUB_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
TOKENHUB_AUDIO_MODEL = _secret("TOKENHUB_AUDIO_MODEL", DEFAULT_AUDIO_MODEL) or DEFAULT_AUDIO_MODEL
TOKENHUB_VIDEO_MODEL = _secret("TOKENHUB_VIDEO_MODEL", DEFAULT_VIDEO_MODEL) or DEFAULT_VIDEO_MODEL
TOKENHUB_BASE_URL = _secret("TOKENHUB_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
PUBLIC_APP_URL = _secret("PUBLIC_APP_URL", "") or ""
AI_CONNECTED = bool(TOKENHUB_API_KEY)


def _accepts_kwarg(func, name: str) -> bool:
    """Return True when the loaded backend helper accepts a keyword.

    Streamlit Cloud can briefly run a new frontend against cached/stale backend
    modules during deployment. Keeping the UI tolerant of older helper signatures
    prevents a visible TypeError while the synchronized package rolls out.
    """
    try:
        params = inspect.signature(func).parameters
        return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        return False


DOMAIN_ZH = {
    "account_access": "账户访问", "banking_fraud": "银行卡欺诈", "delivery": "配送",
    "device_support": "设备支持", "event_ticketing": "活动票务", "healthcare_appointment": "医疗预约",
    "hotel_hospitality": "酒店服务", "insurance_claim": "保险理赔", "internet": "家庭网络",
    "marketplace_dispute": "平台交易纠纷", "payment": "支付", "return_refund": "退货退款",
    "rideshare": "网约车", "software_saas": "企业软件 / SaaS", "subscription": "订阅",
    "telecom_mobile": "移动通信", "travel": "旅行", "utilities": "公用事业",
}
DOMAIN_DESCRIPTIONS_ZH = {
    "account_access": "登录、身份验证、账户锁定与恢复。", "banking_fraud": "可疑交易、银行卡安全、争议与补卡。",
    "delivery": "包裹丢失、延误、错误扫描或错投。", "device_support": "联网设备、固件、配对、硬件故障与排查。",
    "event_ticketing": "票券转让、二维码激活、入场权限与检票问题。", "healthcare_appointment": "预约、确认、医生可用时间与排期不一致。",
    "hotel_hospitality": "预订、酒店系统不一致、房型与订单变更。", "insurance_claim": "理赔状态、缺失材料、审核阻塞与下一步要求。",
    "internet": "家庭网络中断、路由器/调制解调器状态与连接诊断。", "marketplace_dispute": "买卖双方纠纷、替换、履约补救与结案。",
    "payment": "支付拒绝、授权失败、重复扣款风险与结账排查。", "return_refund": "退货、退款状态、仓库处理与到账延迟。",
    "rideshare": "行程扣款、预授权、司机/乘客争议与账户调整。", "software_saas": "权限、授权、工作区配置与 SaaS 服务问题。",
    "subscription": "取消、续费、账单状态与周期订阅问题。", "telecom_mobile": "移动套餐、开通、流量限制、网络策略与运营商账户。",
    "travel": "航班变更、出票、重签、订单状态与行程支持。", "utilities": "电表/水表、账单、服务记录与用量不一致。",
}
CHANNEL_ZH = {
    "Text Messages": "文字消息", "Image Upload": "图片上传", "Audio Upload": "音频上传", "Video Upload": "视频上传", "Voice Call": "语音通话", "Video + Voice": "视频 + 语音", "Multimodal Mix": "多模态混合",
}
CHANNEL_HINT_ZH = {
    "Text Messages": "以文字为主的客服场景；客户措辞是主要信号。",
    "Image Upload": "以图片为主的客服场景；本轮主要上传截图或图片证据。",
    "Audio Upload": "以音频为主的客服场景；上传音频由 Hy-ASR 转写，情绪根据转写文本推断。",
    "Video Upload": "以视频为主的客服场景；上传视频由 YT-VITA 分析画面与音轨。",
    "Voice Call": "实时电话风格客服；以通话转写形式展示客户内容。",
    "Video + Voice": "实时视频通话风格客服；视频与语音上下文会一起解释。",
    "Multimodal Mix": "DeepSeek 文本/图像 + Hy-ASR 音频转写 + YT-VITA 视频 + 后台证据，共享到同一个 JSpace。",
}
EMOTION_ZH = {
    "calm":"平静", "neutral":"中性", "curious":"好奇", "hopeful":"有希望", "appreciative":"感谢", "satisfied":"满意", "relieved":"安心",
    "uncertain":"不确定", "confused":"困惑", "anxious":"焦虑", "disappointed":"失望", "frustrated":"沮丧", "angry":"生气", "impatient":"不耐烦",
    "skeptical":"怀疑", "distressed":"难受", "embarrassed":"尴尬",
}
SOURCE_ZH = {"text":"文字", "audio":"音频", "image":"图像", "video":"视频", "backend":"后台", "derived":"推导"}
STATUS_ZH = {"supported":"已支持", "disputed":"有争议", "unresolved":"未解决"}
PHASE_ZH = {"active":"处理中", "resolving":"解决中", "resolved":"已解决", "closing":"收尾中", "ended":"已结束"}
ACTION_ZH = {
    "close_session": "确认客户没有其他问题，感谢客户并自然结束会话。",
    "confirm_resolution": "明确确认问题已经解决，然后询问客户是否还有其他问题。",
    "resolve_conflict": "核对权威系统状态，解释信息不一致，并避免过早宣布问题已经解决。",
    "act_on_root_cause": "根据已确认的根因采取下一步具体修复动作。",
    "avoid_repetition": "不要重复客户已经完成的排查；直接进入新的诊断步骤。",
    "investigate": "保持工单打开，继续检查权威系统状态和具体阻塞原因。",
    "clarify": "只问一个能够推进问题解决的聚焦问题，不让客户重复已提供的信息。",
}

CONCEPT_NAME_ZH = {
    "authoritative_status": "权威系统状态",
    "customer_visible_status": "客户可见状态",
    "customer_belief_status": "客户认为的状态",
    "root_cause": "根本原因",
    "customer_domain": "客服领域",
    "relationship_state": "客户关系状态",
    "customer_emotion": "客户情绪",
    "emotion_intensity": "情绪强度",
    "retry_count": "重试次数",
    "avoid_repeat_action": "避免重复操作",
    "prior_effort": "客户已完成的排查",
    "backend_event": "后台事件",
    "visual_observation": "视觉观察",
    "channel_visual_context": "渠道视觉信息",
    "audio_transcript": "音频转写",
    "video_summary": "视频摘要",
    "video_visible_evidence": "视频可见证据",
    "video_spoken_content": "视频语音内容",
    "image_analysis_status": "图像分析状态",
    "audio_analysis_status": "音频分析状态",
    "video_analysis_status": "视频分析状态",
}

CONCEPT_VALUE_ZH = {
    "resolved": "已解决", "unresolved": "未解决", "supported": "已支持",
    "appears successful": "看起来已成功", "delivered": "已送达", "wifi visible": "Wi-Fi 可见",
    "partial access": "部分访问", "cancellation requested": "已请求取消", "new itinerary visible": "新行程已显示",
    "return completed": "退货已完成", "100% progress": "进度显示 100%", "connected": "已连接",
    "workspace visible": "工作区可见", "adjusted badge": "已显示调整标记", "confirmed": "已确认",
    "active": "有效", "upgrade visible": "升级信息可见", "cancelled": "已取消", "ticket visible": "票券可见",
    "upgrade complete": "升级已完成", "completed": "已完成",
    "merchant category restriction": "商户类别限制",
    "carrier depot exception": "承运商站点异常",
    "neighborhood fiber outage": "社区光纤中断",
    "risk lock requires identity verification": "风险锁定，需要身份验证",
    "cancellation confirmation was never committed": "取消确认未真正写入系统",
    "ticket reissue failed after itinerary change": "行程变更后票务重签失败",
    "refund workflow stalled after warehouse receipt": "仓库收货后退款流程停滞",
    "proof-of-loss document missing": "缺少损失证明文件",
    "firmware authentication loop": "固件认证循环故障",
    "role entitlement propagation failed": "角色权限同步失败",
    "corrected meter read awaiting ledger posting": "更正后的抄表读数尚未入账",
    "provider schedule change not reflected in portal cache": "医生排班变更尚未同步到门户缓存",
    "fraud restriction pending card replacement workflow": "欺诈限制仍在等待换卡流程完成",
    "upgrade failed to sync to property system": "升级信息未同步到酒店系统",
    "authorization hold awaiting automatic release": "预授权冻结仍在等待自动释放",
    "ticket entitlement activation failed": "票券权限激活失败",
    "plan change not propagated to network policy": "套餐变更尚未同步到网络策略",
    "seller remediation task was never fulfilled": "卖家补救任务尚未完成",
    "avoid repeating completed troubleshooting": "不要重复客户已经完成的排查",
    "customer has already completed troubleshooting": "客户已经完成过排查步骤",
    "live visual evidence available": "已有实时视觉证据",
    "analysis unavailable": "分析不可用",
}



def _is_zh() -> bool:
    return st.session_state.get("ui_language", "English") == "中文"


def L(en: str, zh: str) -> str:
    return zh if _is_zh() else en


def display_domain(domain: str) -> str:
    return DOMAIN_ZH.get(domain, domain.replace("_", " ").title()) if _is_zh() else domain.replace("_", " ").title()


def domain_description(domain: str) -> str:
    return DOMAIN_DESCRIPTIONS_ZH.get(domain, "客户服务场景。") if _is_zh() else DOMAIN_DESCRIPTIONS.get(domain, "Customer-service case.")


def display_channel(channel: str) -> str:
    return CHANNEL_ZH.get(channel, channel) if _is_zh() else channel


def channel_hint(channel: str) -> str:
    return CHANNEL_HINT_ZH.get(channel, CHANNELS[channel]["hint"]) if _is_zh() else CHANNELS[channel]["hint"]


def _language_prompt_name() -> str:
    return "Simplified Chinese" if _is_zh() else "English"


def display_concept_name(name: str) -> str:
    if not _is_zh():
        return str(name).replace("_", " ").title()
    if name in CONCEPT_NAME_ZH:
        return CONCEPT_NAME_ZH[name]
    if name.startswith("image_evidence_"):
        suffix = name.rsplit("_", 1)[-1]
        return f"图像证据 {suffix}"
    return str(name).replace("_", " ")


def display_concept_value(name: str, value) -> str:
    raw = str(value)
    if not _is_zh():
        return raw
    low = raw.strip().lower()
    if name == "customer_domain":
        return DOMAIN_ZH.get(low, raw)
    if name == "customer_emotion":
        return EMOTION_ZH.get(low, raw)
    if name == "relationship_state":
        return {"new":"新客户", "positive":"良好", "loyal":"忠诚", "neutral":"中性", "strained":"紧张", "at risk":"流失风险"}.get(low, raw)
    return CONCEPT_VALUE_ZH.get(low, raw)


def manual_mode_config(channel: str) -> dict:
    return MANUAL_MODE_CONFIG.get(channel, MANUAL_MODE_CONFIG["Text Messages"])


STATUS_CONCEPT_NAMES = {"authoritative_status", "customer_visible_status", "customer_belief_status"}


def ordered_active_concepts(state) -> list:
    concepts = list(getattr(state, "active_concepts", []) or [])
    return sorted(concepts, key=lambda c: ((c.name in STATUS_CONCEPT_NAMES), -getattr(c, "score", 0.0), c.name))


def primary_workspace_concepts(state) -> list:
    """Show task concepts first without letting routine status cards dominate the UI.

    Status concepts remain available in the reserved status lane below and in provenance.
    If status concepts consumed active Top-K slots, fill the visual task lane with the
    highest-ranked non-status candidates so researchers can inspect a richer concept set.
    """
    capacity = int(getattr(getattr(state, "config", None), "capacity_k", 4) or 4)
    active = [c for c in getattr(state, "active_concepts", []) if c.name not in STATUS_CONCEPT_NAMES]
    seen = {c.name for c in active}
    extras = sorted(
        [c for c in getattr(state, "concepts", []) if c.name not in STATUS_CONCEPT_NAMES and c.name not in seen],
        key=lambda c: getattr(c, "score", 0.0), reverse=True,
    )
    return (active + extras)[:capacity]


def status_workspace_concepts(state) -> list:
    by_name = {}
    for c in list(getattr(state, "concepts", []) or []) + list(getattr(state, "active_concepts", []) or []):
        if c.name in STATUS_CONCEPT_NAMES:
            by_name[c.name] = c
    return sorted(by_name.values(), key=lambda c: c.name)


def localize_generated_scenario(scenario):
    """Guarantee Chinese customer-facing scenario text when the global UI is Chinese.

    DeepSeek normally rewrites the scenario first. If that provider rewrite fails, this
    deterministic fallback still keeps every customer turn in Chinese rather than
    leaking the English curated template into a Chinese session.
    """
    if not _is_zh():
        return scenario
    updated = scenario.model_copy(deep=True)
    starter = CUSTOMER_STARTERS_ZH.get(
        updated.domain,
        "我遇到了一个客服问题。你能先帮我确认一下当前系统状态吗？",
    )
    fallback_by_label = {
        "Opening issue": starter,
        "Customer explains impact": "这个问题已经影响到我现在的使用了。你能告诉我为什么还没有恢复正常吗？",
        "Customer adds prior context": "我已经按照之前的建议尝试过了，但问题还是存在。你能不要让我重复同样的步骤，直接继续往下查吗？",
        "Customer sees apparently conflicting evidence": "我这边看到的状态和你们系统说的不一样。现在到底哪个状态才是准确的？",
        "Diagnostic result becomes available": "你目前查到了什么？我最想知道的是具体哪个环节还在阻塞。",
        "Additional troubleshooting context 1": "你能再看看历史记录里有没有其他信息，可以解释为什么会发生这个问题吗？",
        "Additional troubleshooting context 2": "如果修复真的生效了，我这边应该看到什么变化？",
        "Additional troubleshooting context 3": "为了把这个问题彻底处理完，你还需要我提供什么信息吗？",
        "Customer asks for resolution details": "在结束之前，你能告诉我这次具体会怎么解决，以及之后我需要注意什么吗？",
        "Resolution confirmed": "我这边现在看起来已经正常了。你能再确认一下系统里也已经完全解决了吗？",
        "No other concerns": "没有其他问题了，谢谢你的帮助。",
    }
    if sum('\u4e00' <= ch <= '\u9fff' for ch in str(updated.title)) < 3:
        updated.title = "中文客服冲突解决场景"
    if sum('\u4e00' <= ch <= '\u9fff' for ch in str(updated.problem_summary or "")) < 6:
        updated.problem_summary = "客户看到的信息与公司系统记录存在差异，需要客服逐步核实、解释冲突并确认最终解决。"
    for i, step in enumerate(updated.steps):
        if not any('\u4e00' <= ch <= '\u9fff' for ch in str(step.customer_turn.text)):
            step.customer_turn.text = fallback_by_label.get(
                step.label,
                "你能根据刚才查到的新信息告诉我下一步具体应该怎么推进吗？",
            )
    return updated


SETTINGS_DEFAULTS = {
    "speed": "Fast",
    "reply": "Concise",
    "scenario_ai": True,
    "auto_scroll": True,
    "researcher_view": False,
}


def _init_preferences() -> None:
    # Keep settings in a non-widget state object. Streamlit may clean widget keys when
    # a dialog is not rendered, but this persistent object survives tab changes/reruns.
    st.session_state.setdefault("app_settings", dict(SETTINGS_DEFAULTS))
    for name, default in SETTINGS_DEFAULTS.items():
        legacy_key = f"settings_{name}"
        if legacy_key in st.session_state:
            st.session_state["app_settings"][name] = st.session_state[legacy_key]
        st.session_state["app_settings"].setdefault(name, default)
    st.session_state.setdefault("ui_language", "English")
    st.session_state.setdefault("generation_epoch", 0)


def _setting(name: str):
    return st.session_state.get("app_settings", SETTINGS_DEFAULTS).get(name, SETTINGS_DEFAULTS[name])


def _persist_setting(widget_key: str, name: str) -> None:
    st.session_state.setdefault("app_settings", dict(SETTINGS_DEFAULTS))
    st.session_state["app_settings"][name] = st.session_state.get(widget_key, SETTINGS_DEFAULTS[name])


def _seed_setting_widget(widget_key: str, name: str) -> None:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = _setting(name)


def _ai_runtime() -> dict:
    speed = _setting("speed")
    reply = _setting("reply")
    if speed == "Fast":
        # Bound TokenHub calls explicitly and disable hidden SDK retries.
        # DeepSeek streams can return much sooner; this is only the per-attempt cap.
        timeout_ms, attempts, history = 12000, 2, 4
    else:
        timeout_ms, attempts, history = 20000, 2, 6
    if reply == "Concise":
        output_tokens, sentences = 120, 2
    else:
        output_tokens, sentences = 180, 3
    return {
        "timeout_ms": timeout_ms,
        "max_attempts": attempts,
        "history_turns": history,
        "max_output_tokens": output_tokens,
        "reply_sentences": sentences,
        "scenario_timeout_ms": 12000 if speed == "Fast" else 20000,
    }


def _bump_generation_epoch() -> None:
    st.session_state.generation_epoch = int(st.session_state.get("generation_epoch", 0)) + 1


def _on_main_tab_change() -> None:
    # Invalidate any UI generation result from the tab the user just left.
    _bump_generation_epoch()


def _scroll_to(element_id: str) -> None:
    components.html(
        f"""<script>setTimeout(function(){{const el=window.parent.document.getElementById('{element_id}'); if(el) el.scrollIntoView({{behavior:'smooth',block:'start'}});}},120);</script>""",
        height=0,
    )


def _scroll_latest_conversation() -> None:
    components.html(
        """<script>setTimeout(function(){const xs=window.parent.document.querySelectorAll('.j-phone'); xs.forEach(function(el){if(el.offsetParent!==null){el.scrollTop=el.scrollHeight;}});},90);</script>""",
        height=0,
    )


_init_preferences()

if st.session_state.pop("language_notice_pending", False):
    st.toast(L("Language changed. Practice sessions restarted so generated messages stay consistent.", "语言已切换。练习会话已重新开始，以保证生成消息语言一致。"), icon="🌐")

# Disable Streamlit's plain R/C developer shortcuts while preserving normal typing
# inside inputs and standard Ctrl/Cmd keyboard shortcuts.
components.html(
    """
<script>
(function(){
  const parentWin = window.parent;
  const doc = parentWin.document;
  if (parentWin.__jspaceShortcutGuardInstalled) return;
  parentWin.__jspaceShortcutGuardInstalled = true;
  const guard = function(e){
    const tag = (e.target && e.target.tagName ? e.target.tagName : '').toUpperCase();
    const editable = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable);
    const plainKey = !e.ctrlKey && !e.metaKey && !e.altKey;
    if (!editable && plainKey && (e.key === 'r' || e.key === 'R' || e.key === 'c' || e.key === 'C')) {
      e.preventDefault();
      e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    }
  };
  doc.addEventListener('keydown', guard, true);
  parentWin.addEventListener('keydown', guard, true);
})();
</script>
""",
    height=0,
)


def reset_sessions() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("live_", "manual_")):
            del st.session_state[key]


# Compact utility controls in the top-right. Modal dialogs keep the toolbar
# minimal; a dedicated language toggle changes the whole customer-facing UI.
@st.dialog("Help" if not _is_zh() else "帮助", width="small")
def _help_dialog() -> None:
    st.markdown(L("**Quick guide**", "**快速指南**"))
    st.write(L(
        "Scenario Lab generates a case and reveals it turn by turn. Manual mode lets you play the customer until you end the session.",
        "Scenario Lab 会生成一个客服案例并逐轮展开。Manual 模式让你扮演客户，可以一直对话到你主动结束会话。",
    ))
    st.write(L(
        "Multimodal Mix routes text/images to DeepSeek Vision, audio to Hy-ASR transcription, video to YT-VITA, then combines the evidence with company-system data.",
        "多模态混合模式会把文字/图片交给 DeepSeek Vision，把音频交给 Hy-ASR 转写，把视频交给 YT-VITA，再与公司后台证据一起进入 JSpace。",
    ))


def _copy_current_link() -> None:
    """Copy the live app URL to the clipboard; no email/share workflow."""
    components.html(
        """<script>
        (async function(){
          try {
            const url = window.parent.location.href;
            await window.parent.navigator.clipboard.writeText(url);
          } catch (e) {
            try {
              const url = window.parent.location.href;
              const ta = window.parent.document.createElement('textarea');
              ta.value = url; ta.style.position='fixed'; ta.style.opacity='0';
              window.parent.document.body.appendChild(ta); ta.select();
              window.parent.document.execCommand('copy'); ta.remove();
            } catch (_) {}
          }
        })();
        </script>""",
        height=0,
    )
    st.toast(L("Link copied to clipboard", "链接已复制到剪贴板"), icon="✅")


@st.dialog("Settings" if not _is_zh() else "设置", width="small")
def _settings_dialog() -> None:
    st.markdown(L("**Conversation settings**", "**对话设置**"))
    for widget_key, name in [
        ("settings_speed", "speed"), ("settings_reply", "reply"),
        ("settings_scenario_ai", "scenario_ai"), ("settings_auto_scroll", "auto_scroll"),
        ("settings_researcher_view", "researcher_view"),
    ]:
        _seed_setting_widget(widget_key, name)
    st.selectbox(
        L("AI response profile", "AI 响应速度"), ["Fast", "Balanced"], key="settings_speed",
        on_change=_persist_setting, args=("settings_speed", "speed"),
        help=L("Fast uses a 12-second DeepSeek attempt cap and 4 recent messages. Balanced uses a 20-second cap and 6 recent messages.", "Fast：单次 DeepSeek 最长 12 秒、保留最近 4 条消息；Balanced：最长 20 秒、保留最近 6 条消息。"),
    )
    st.selectbox(L("Agent reply length", "客服回复长度"), ["Concise", "Standard"], key="settings_reply", on_change=_persist_setting, args=("settings_reply", "reply"))
    st.toggle(L("Use DeepSeek to vary scenario wording", "使用 DeepSeek 改写场景措辞"), key="settings_scenario_ai", on_change=_persist_setting, args=("settings_scenario_ai", "scenario_ai"), help=L("Turn this off for near-instant curated scenario generation.", "关闭后会直接使用预设场景，生成速度更快。"))
    st.toggle(L("Auto-scroll conversations", "对话自动滚动到底部"), key="settings_auto_scroll", on_change=_persist_setting, args=("settings_auto_scroll", "auto_scroll"))
    st.toggle(L("Enable Researcher View", "启用研究者视图"), key="settings_researcher_view", on_change=_persist_setting, args=("settings_researcher_view", "researcher_view"), help=L("Shows hidden simulated ground truth and provider diagnostics. Off by default.", "显示隐藏的模拟真值和模型诊断信息。默认关闭。"))
    cfg = _ai_runtime()
    st.caption(L(
        f"Text/image: {TOKENHUB_MODEL} · Audio: {TOKENHUB_AUDIO_MODEL} · Video: {TOKENHUB_VIDEO_MODEL}",
        f"文字/图像：{TOKENHUB_MODEL} · 音频：{TOKENHUB_AUDIO_MODEL} · 视频：{TOKENHUB_VIDEO_MODEL}",
    ))
    st.caption(L(
        f"DeepSeek: {cfg['timeout_ms']/1000:.0f}s/attempt · {cfg['history_turns']} recent messages · max {cfg['max_attempts']} attempts",
        f"DeepSeek：每次最多 {cfg['timeout_ms']/1000:.0f} 秒 · 最近 {cfg['history_turns']} 条消息 · 最多 {cfg['max_attempts']} 次尝试",
    ))
    if st.button(L("Test DeepSeek connection", "测试 DeepSeek 连接"), width="stretch", key="test_deepseek"):
        with st.spinner(L("Testing…", "正在测试…")):
            ok, detail = probe_deepseek(api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL, timeout_s=12.0)
        if ok:
            st.success(L(f"Connected · {TOKENHUB_MODEL}", f"连接成功 · {TOKENHUB_MODEL}"))
        else:
            st.error(detail)
    components.html(
        f"""<button onclick="window.parent.print()" style="width:100%;padding:8px 12px;border-radius:10px;border:1px solid #4a7891;background:#0d1b2b;color:#eaf7ff;cursor:pointer">{L('Print this view', '打印当前页面')}</button>""",
        height=46,
    )


def _toggle_language() -> None:
    st.session_state.ui_language = "中文" if st.session_state.get("ui_language", "English") == "English" else "English"
    st.session_state.language_notice_pending = True
    _bump_generation_epoch()
    reset_sessions()


with st.container(key="utility_toolbar"):
    # Stable normal-flow header: the large first column pushes controls right without
    # detaching Streamlit's grid from page layout.
    spacer, lang_col, u1, u2, u3, u4 = st.columns([12.0, 1.15, .72, .72, .72, .72], gap="small", vertical_alignment="center")
    with lang_col:
        st.button(
            "中文" if not _is_zh() else "EN",
            help=L("Switch the entire experience to Chinese. Active practice sessions restart so generated customer/agent text stays in one language.", "切换整页为英文。当前练习会话会重新开始，以保证客户和 AI 消息保持同一种语言。"),
            key="top_language", width="stretch", on_click=_toggle_language,
        )
    with u1:
        if st.button(" ", icon=":material/help:", help=L("Help", "帮助"), key="top_help", width="stretch"):
            _help_dialog()
    with u2:
        if st.button(" ", icon=":material/link:", help=L("Copy page link", "复制页面链接"), key="top_share", width="stretch"):
            _copy_current_link()
    with u3:
        if st.button(" ", icon=":material/refresh:", help=L("Reset current sessions", "重置当前会话"), key="top_reset", width="stretch"):
            _bump_generation_epoch()
            reset_sessions()
            st.rerun()
    with u4:
        if st.button(" ", icon=":material/settings:", help=L("Settings", "设置"), key="top_settings", width="stretch"):
            _settings_dialog()

st.markdown("<div style=\"height:.65rem\"></div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="j-hero">
  <div class="j-kicker">{L("JSPACE // MULTIMODAL SERVICE LAB", "JSPACE // 多模态客服实验室")}</div>
  <div class="j-title">{L("Live customer-service reasoning across text, voice, video and system evidence.", "跨文字、语音、视频与系统证据的实时客服推理。")}</div>
  <div class="j-sub">{L("Observe a capacity-limited shared workspace preserve task-critical signals, uncertainty and cross-modal conflicts while the support agent works toward a natural resolution.", "观察一个容量受限的共享工作空间如何保留关键任务信号、不确定性和跨模态冲突，并推动客服自然地解决问题。")}</div>
  <span class="j-pill">v{APP_VERSION}</span>
  <span class="j-pill">{L("Tencent multimodal routing enabled", "腾讯多模态路由已启用") if AI_CONNECTED else L("Local simulation mode", "本地模拟模式")}</span>
</div>
""",
    unsafe_allow_html=True,
)



PROFILE_VALUE_ZH = {
    "new customer":"新客户", "6 months":"6 个月", "new":"新客户", "positive":"良好", "loyal":"忠诚", "neutral":"一般", "strained":"紧张", "at risk":"流失风险",
    "standard":"标准", "silver":"银卡", "gold":"金卡", "platinum":"白金", "high value":"高价值", "strategic":"战略客户", "occasional":"偶尔使用",
    "concise":"简洁", "detail-oriented":"注重细节", "conversational":"对话型", "direct":"直接", "cautious":"谨慎", "question-heavy":"偏多提问",
    "low":"低", "medium":"中", "high":"高", "voice":"语音", "chat":"聊天", "mobile app":"移动应用", "email":"邮件",
}


def _display_profile_value(key: str, value) -> str:
    raw = str(value)
    if not _is_zh():
        return raw
    if key == "tenure":
        import re as _re
        match = _re.fullmatch(r"(\d+) years?", raw)
        if match:
            return f"{match.group(1)} 年"
    return PROFILE_VALUE_ZH.get(raw.lower(), raw)


def profile_html(profile: dict) -> str:
    items = [
        (L("Customer", "客户"), profile.get("name", "—")), (L("Tenure", "客户年限"), _display_profile_value("tenure", profile.get("tenure", "—"))),
        (L("Relationship", "关系状态"), _display_profile_value("relationship", profile.get("relationship", "—"))), (L("Loyalty", "忠诚度"), _display_profile_value("loyalty", profile.get("loyalty_tier", "—"))),
        (L("Contacts · 90d", "90 天联系次数"), profile.get("previous_contacts_90d", "—")), (L("Value segment", "价值分层"), _display_profile_value("value", profile.get("value_segment", "—"))),
        (L("Communication", "沟通风格"), _display_profile_value("communication", profile.get("communication_style", "—"))), (L("Tech comfort", "技术熟悉度"), _display_profile_value("tech", profile.get("tech_comfort", "—"))),
    ]
    cells = "".join(
        f'<div class="j-profile-cell"><div class="j-profile-label">{html.escape(str(k))}</div><div class="j-profile-value">{html.escape(str(v))}</div></div>'
        for k, v in items
    )
    return f'<div class="j-profile-grid">{cells}</div>'


def render_profile(profile: dict, state=None) -> None:
    st.markdown(profile_html(profile), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.progress(max(0.0, min(1.0, profile.get("patience", 0) / 100)))
    c1.caption(f"{L('Patience', '耐心度')} · {max(0, int(profile.get('patience', 0)))}/100")
    c2.progress(max(0.0, min(1.0, profile.get("trust", 0) / 100)))
    c2.caption(f"{L('Trust in company', '对公司的信任')} · {profile.get('trust', 0)}/100")
    satisfaction = float(getattr(state, "customer_satisfaction", 50.0)) if state is not None else 50.0
    c3.progress(max(0.0, min(1.0, satisfaction / 100)))
    c3.caption(f"{L('Satisfaction', '满意度')} · {satisfaction:.0f}/100")
    c4.markdown(
        f'<div class="j-card"><div class="j-card-title">{L("Preferred channel", "偏好渠道")}</div><div class="j-card-value">{html.escape(_display_profile_value("preferred_channel", profile.get("preferred_channel", "—")))}</div></div>',
        unsafe_allow_html=True,
    )


def _emotion_html(state) -> str:
    label = (EMOTION_ZH.get(state.current_emotion, "等待信号") if _is_zh() else (state.current_emotion or "Waiting for signal").replace("_", " ").title())
    size = "1.42rem" if len(label) <= 11 else ("1.14rem" if len(label) <= 17 else ".96rem")
    intensity = f"{state.current_emotion_intensity:.0%}" if state.current_emotion else "—"
    return f'''<div class="j-emotion"><div class="j-emotion-label">{L("Customer affect", "客户情绪")}</div><div class="j-emotion-value" style="font-size:{size}">{html.escape(label)}</div><div class="j-card-meta">{L("Affect intensity", "情绪强度")} · {intensity}</div></div>'''


def concept_rows(state) -> pd.DataFrame:
    rows = []
    for c in ordered_active_concepts(state):
        rows.append({
            L("Concept", "概念"): display_concept_name(c.name),
            L("Value", "值"): display_concept_value(c.name, c.value),
            L("Status", "状态"): STATUS_ZH.get(c.status, c.status) if _is_zh() else c.status,
            L("Sources", "来源"): ", ".join(SOURCE_ZH.get(src, src) if _is_zh() else src for src in c.sources),
            L("Priority", "优先级"): round(c.score, 2),
            L("Confidence", "置信度"): round(c.confidence, 2),
        })
    return pd.DataFrame(rows)



CONFLICT_SEVERITY_LABELS = {
    "high": ("HIGH PRIORITY SIGNAL CONFLICT", "高优先级信号冲突"),
    "medium": ("MEDIUM PRIORITY SIGNAL CONFLICT", "中优先级信号冲突"),
    "low": ("LOW PRIORITY SIGNAL CONFLICT", "低优先级信号冲突"),
}


def display_conflict_severity(severity: str) -> str:
    key = str(severity or "medium").strip().lower()
    en, zh = CONFLICT_SEVERITY_LABELS.get(key, (f"{key.upper()} PRIORITY SIGNAL CONFLICT", f"{key}优先级信号冲突"))
    return zh if _is_zh() else en


def display_conflict_description(conflict) -> str:
    """Show the conflict explanation in the active UI language.

    Chinese mode also localizes stale Conflict objects created before
    `description_zh` existed, so a hot-reloaded/redeployed session cannot keep
    showing an English paragraph inside an otherwise Chinese conflict card.
    """
    raw = str(getattr(conflict, "description", "") or "")
    if _is_zh():
        return conflict_description_zh(raw, getattr(conflict, "description_zh", ""))
    return raw


def render_workspace(state, *, show_coaching: bool = True) -> None:
    st.markdown(L("#### Live JSpace", "#### 实时 JSpace"))
    if show_coaching:
        top1, top2 = st.columns([1.35, .65])
        with top1:
            st.markdown(
                f'''<div class="j-card j-next"><div class="j-card-title">{L("Recommended next move", "推荐下一步")}</div><div class="j-card-value">{html.escape((ACTION_ZH.get(state.recommended_action_code, state.recommended_action or "先获取客户的第一个有效信号") if _is_zh() else (state.recommended_action or "Gather the first customer signal")))}</div><div class="j-card-meta">{L("Support coaching cue · the next action most likely to advance resolution.", "客服辅助提示 · 最有可能推进问题解决的下一步动作。")}</div></div>''',
                unsafe_allow_html=True,
            )
        with top2:
            st.markdown(_emotion_html(state), unsafe_allow_html=True)
    else:
        st.markdown(_emotion_html(state), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.caption(f"{L('Active concepts', '活跃概念')} · {len(ordered_active_concepts(state))} / {state.config.capacity_k}")
    c2.caption(f"{L('Signal conflicts', '信号冲突')} · {len(state.conflicts)}")
    c3.caption(f"{L('Session phase', '会话阶段')} · {(PHASE_ZH.get(state.session_phase, state.session_phase) if _is_zh() else state.session_phase.title())}")

    if state.conflicts:
        for conflict in state.conflicts:
            st.markdown(
                f'<div class="j-card j-conflict"><div class="j-card-title">{html.escape(display_conflict_severity(conflict.severity))}</div><div class="j-card-value">{html.escape(display_conflict_description(conflict))}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(L("##### Primary task concepts", "##### 主要任务概念"))
    display_concepts = primary_workspace_concepts(state)
    if not display_concepts:
        st.info(L("JSpace will populate as customer, media, and company evidence arrive.", "随着客户、媒体和公司系统证据进入，JSpace 会逐步填充。"))
    else:
        for c in display_concepts:
            sources = " · ".join((SOURCE_ZH.get(src, src) if _is_zh() else src.title()) for src in c.sources)
            st.markdown(
                f'''<div class="j-card j-concept {html.escape(c.status)}"><div class="j-card-title">{html.escape(display_concept_name(c.name))}</div><div class="j-card-value">{html.escape(display_concept_value(c.name, c.value))}</div><div class="j-card-meta">{html.escape(sources)} · {L('priority', '优先级')} {c.score:.2f} · {html.escape(STATUS_ZH.get(c.status, c.status) if _is_zh() else c.status)}</div></div>''',
                unsafe_allow_html=True,
            )

    status_concepts = status_workspace_concepts(state)
    if status_concepts:
        with st.expander(L("Resolution/status context", "解决状态上下文"), expanded=False):
            st.caption(L("Status remains available for conflict reasoning but no longer dominates the primary concept display.", "状态信息仍会参与冲突推理，但不会再占据主要概念展示的顶部。"))
            for c in status_concepts:
                sources = " · ".join((SOURCE_ZH.get(src, src) if _is_zh() else src.title()) for src in c.sources)
                st.markdown(
                    f'''<div class="j-card j-concept {html.escape(c.status)}"><div class="j-card-title">{html.escape(display_concept_name(c.name))}</div><div class="j-card-value">{html.escape(display_concept_value(c.name, c.value))}</div><div class="j-card-meta">{html.escape(sources)} · {L('priority', '优先级')} {c.score:.2f}</div></div>''',
                    unsafe_allow_html=True,
                )

    with st.expander(L("Evidence & provenance", "证据与来源"), expanded=False):
        df = concept_rows(state)
        if df.empty:
            st.caption(L("No active evidence yet.", "暂时没有活跃证据。"))
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(L("Priority is a workspace ranking signal, not an accuracy or satisfaction score.", "Priority 是工作空间排序信号，并不是准确率或满意度分数。"))


def _message_html(row: dict, channel_label: str) -> str:
    role = row.get("role", "customer")
    text = row.get("text", "")
    if role == "agent":
        who = L("JSpace Support Agent", "JSpace 客服")
        provider = str(row.get("provider") or "").strip()
        if provider:
            if provider.lower().startswith("deepseek"):
                meta = L("AI provider · ", "AI 提供方 · ") + provider
            elif "fallback" in provider.lower() or "simulation" in provider.lower():
                meta = L("Backup responder · ", "备用响应器 · ") + provider
            else:
                meta = provider
        else:
            meta = ""
    else:
        who = L("Customer", "客户")
        details = []
        if row.get("emotion"):
            details.append(EMOTION_ZH.get(str(row["emotion"]), str(row["emotion"])) if _is_zh() else str(row["emotion"]).replace("_", " ").title())
        if isinstance(row.get("emotion_intensity"), (float, int)):
            details.append(f"{row['emotion_intensity']:.0%} {L('affect', '情绪强度')}")
        if row.get("nonverbal_cue") and channel_label != "Text Messages":
            details.append(str(row["nonverbal_cue"]))
        attachments = row.get("attachments", [])
        if attachments:
            details.append(L("media: ", "媒体：") + ", ".join(x.get("name", L("attachment", "附件")) for x in attachments))
        meta = " · ".join(details)
    meta_html = f'<div class="j-msg-meta">{html.escape(str(meta))}</div>' if meta else ""
    return f'''<div class="j-msg-row {role}"><div class="j-msg {role}"><div style="font-size:.63rem;color:#81A5C5;font-weight:700;margin-bottom:.24rem">{html.escape(who)}</div>{html.escape(str(text))}{meta_html}</div></div>'''


def _conversation_html(transcript: list[dict], channel_label: str, *, typing: bool = False) -> str:
    info = CHANNELS[channel_label]
    messages = [_message_html(row, channel_label) for row in transcript]
    if typing:
        messages.append(f'<div class="j-msg-row agent"><div class="j-msg agent"><div class="j-typing">{L("Support Agent is typing…", "客服正在输入…")}</div></div></div>')
    if not messages:
        messages.append(f'<div class="j-card-meta" style="padding:.9rem">{L("Conversation has not started yet.", "会话尚未开始。")}</div>')
    return f'''<div class="j-phone"><div class="j-phone-head"><div><div class="j-channel-name">{info['icon']} {html.escape(display_channel(channel_label))}</div><div class="j-channel-meta">{html.escape(channel_hint(channel_label))}</div></div><div class="j-channel-meta"><span class="j-live-dot"></span>{L("LIVE SESSION", "实时会话")}</div></div>{''.join(messages)}</div>'''


def render_conversation(transcript: list[dict], channel_label: str, *, typing: bool = False, slot=None) -> None:
    target = slot or st.empty()
    target.markdown(_conversation_html(transcript, channel_label, typing=typing), unsafe_allow_html=True)
    if _setting("auto_scroll") and transcript:
        _scroll_latest_conversation()


def make_responder(channel_label: str, media: list[dict] | None = None):
    def responder(state, profile, domain):
        cfg = _ai_runtime()
        kwargs = dict(
            api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL,
            fallback=state.last_response, channel=CHANNELS[channel_label]["slug"], media=media,
            timeout_s=cfg["timeout_ms"] / 1000.0, max_attempts=cfg["max_attempts"],
            history_turns=cfg["history_turns"], max_output_tokens=cfg["max_output_tokens"],
            reply_sentences=cfg["reply_sentences"],
        )
        if _accepts_kwarg(generate_support_reply, "language"):
            kwargs["language"] = _language_prompt_name()
        return generate_support_reply(state, profile, domain, **kwargs)
    return responder


def stream_agent_reply(state, profile, domain: str, channel_label: str, conversation_slot, *, media=None, epoch: int | None = None):
    """Stream the support reply into the existing phone UI and return the final text/provider."""
    cfg = _ai_runtime()
    final_text, final_provider = state.last_response or L("I can help with that.", "我可以帮你处理这个问题。"), "Local simulation"
    got_text = False
    stream_kwargs = dict(
        api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL,
        fallback=state.last_response, channel=CHANNELS[channel_label]["slug"], media=media,
        timeout_s=cfg["timeout_ms"] / 1000.0, max_attempts=cfg["max_attempts"],
        history_turns=cfg["history_turns"], max_output_tokens=cfg["max_output_tokens"],
        reply_sentences=cfg["reply_sentences"],
    )
    if _accepts_kwarg(stream_support_reply, "language"):
        stream_kwargs["language"] = _language_prompt_name()
    for partial, provider, done in stream_support_reply(state, profile, domain, **stream_kwargs):
        if epoch is not None and int(st.session_state.get("generation_epoch", 0)) != epoch:
            return None, "Canceled"
        final_text, final_provider = partial, provider
        if partial:
            got_text = True
            preview = state.transcript + [{"role": "agent", "text": partial, "provider": provider}]
            render_conversation(preview, channel_label, slot=conversation_slot)
        if done:
            break
    if not got_text:
        render_conversation(state.transcript, channel_label, typing=True, slot=conversation_slot)
    return final_text, final_provider


def read_uploaded_media(files) -> tuple[list[dict], list[dict]]:
    full, display = [], []
    for f in files or []:
        data = f.getvalue()
        mime = f.type or "application/octet-stream"
        full.append({"name": f.name, "mime_type": mime, "data": data})
        display.append({"name": f.name, "mime_type": mime})
    return full, display


def prepare_scenario_for_channel(scenario, channel_label: str):
    scenario = scenario.model_copy(deep=True)
    affect_source = CHANNELS[channel_label]["affect_source"]
    for step in scenario.steps:
        step.customer_turn.affect_source = affect_source
        if channel_label == "Text Messages":
            step.customer_turn.nonverbal_cue = None
        elif channel_label == "Voice Call":
            # Voice mode emphasizes vocal affect rather than visual evidence.
            step.image_observations = []

    if channel_label in {"Video + Voice", "Multimodal Mix"} and len(scenario.steps) >= 3:
        idx = min(2, len(scenario.steps) - 1)
        extra = ImageObservation(
            description=(
                "Live visual evidence from the customer shows the current app/device/service state while they explain the issue"
                if channel_label == "Video + Voice"
                else "Customer-provided visual evidence adds an independent modality that may support or challenge the spoken/text account"
            ),
            concept_name="channel_visual_context",
            concept_value="live visual evidence available",
            confidence=0.84,
            relevance=0.68,
            conflict_importance=0.36,
        )
        scenario.steps[idx].image_observations.append(extra)
        scenario.hidden_ground_truth["channel_feature"] = "visual + affect + backend evidence enabled"
    elif channel_label == "Voice Call":
        scenario.hidden_ground_truth["channel_feature"] = "audio affect + backend evidence enabled"
    else:
        scenario.hidden_ground_truth["channel_feature"] = "text + optional visual evidence"
    return scenario


def _manual_ready_to_close(state) -> bool:
    """Return True when the manual customer can naturally close instead of asking again."""
    if getattr(state, "session_ended", False):
        return False
    if getattr(state, "session_phase", "active") in {"resolved", "closing"}:
        return True
    if getattr(state, "conflicts", []):
        return False
    satisfaction = float(getattr(state, "customer_satisfaction", 0.0) or 0.0)
    action = str(getattr(state, "recommended_action_code", "") or "")
    last_agent = next((str(r.get("text") or "") for r in reversed(getattr(state, "transcript", [])) if r.get("role") == "agent"), "")
    low = last_agent.lower()
    resolution_language = any(token in low for token in [
        "resolved", "fixed", "working now", "all set", "completed", "restored",
        "anything else", "other questions", "other concerns",
        "已经解决", "已解决", "恢复正常", "处理完成", "已经恢复", "还有其他", "其他问题",
    ])
    return satisfaction >= 76 and (action in {"confirm_resolution", "close_session"} or resolution_language)


def _unused_customer_move(state, candidates: list[str]) -> str:
    """Choose an unused natural customer line without exposing simulation mechanics."""
    used = {str(r.get("text") or "").strip() for r in getattr(state, "transcript", []) if r.get("role") == "customer"}
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in used:
            return candidate
    # Never fall back to simulation meta-language. If a small candidate bank is
    # exhausted, use a plain customer reply instead.
    natural_fallbacks = [
        L("Okay. Please just get this sorted out for me.", "好的，请直接帮我把这个问题处理好。"),
        L("Understood. I just need the issue taken care of now.", "明白。我现在只需要把这个问题解决掉。"),
        L("Thanks. Please go ahead with the fix.", "谢谢，请直接继续修复。"),
    ]
    for candidate in natural_fallbacks:
        if candidate not in used:
            return candidate
    return natural_fallbacks[0]


def _varied_customer_move(state, candidates: list[str], salt: str) -> str:
    """Rotate a natural candidate bank by session/turn so practices do not feel scripted."""
    candidates = [str(x).strip() for x in candidates if str(x or "").strip()]
    if not candidates:
        return ""
    customer_turns = sum(1 for r in getattr(state, "transcript", []) if r.get("role") == "customer")
    token = f"{getattr(state, 'session_id', '')}:{customer_turns}:{salt}"
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(token)) % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    return _unused_customer_move(state, ordered)


def suggested_customer_prompt(domain: str, state) -> str:
    """Suggest the next line as a manual version of Scenario Lab.

    The customer moves through the same realistic arc: issue -> impact -> prior context
    -> diagnosis -> authorize the fix -> acknowledge resolution -> close.  Suggestions
    are customer dialogue only; they never mention turn numbers, prompts, verification
    mechanics, or ask the agent to narrate its internal workflow.
    """
    case = getattr(state, "manual_context", {}) or {}
    customer_turns = sum(1 for r in getattr(state, "transcript", []) if r.get("role") == "customer")
    if customer_turns == 0:
        opening = str(case.get("opening") or "").strip()
        if opening:
            return opening
        return (CUSTOMER_STARTERS_ZH if _is_zh() else CUSTOMER_STARTERS).get(
            domain, L("I'm having an issue with my account and need some help.", "我的账户出了点问题，想请你帮我处理一下。")
        )
    if state.session_ended:
        return ""

    # Resolution is a terminal state in the simulation.  The next customer move is
    # gratitude/closure, not another request to confirm what was already confirmed.
    if _manual_ready_to_close(state):
        closings = list(case.get("closings") or [])
        if _is_zh():
            closings = []
        return _varied_customer_move(state, closings + [
            L("No, that's everything. Thanks for getting it sorted out.", "没有其他问题了。谢谢你帮我处理好。"),
            L("That's all I needed. Thank you for fixing it.", "这就是我需要的。谢谢你帮我修复。"),
            L("No other questions. I appreciate the help.", "没有其他问题了，谢谢你的帮助。"),
        ], "closing")

    authoritative = next((c for c in getattr(state, "concepts", []) if getattr(c, "name", "") == "authoritative_status"), None)
    authoritative_unresolved = bool(authoritative and str(getattr(authoritative, "value", "")).lower() != "resolved")
    has_root = any(getattr(c, "name", "") == "root_cause" for c in getattr(state, "concepts", []))

    # These first turns come directly from the generated manual case, just like the
    # first turns in Scenario Lab.
    if customer_turns == 1:
        return _varied_customer_move(state, [
            str(case.get("impact") or ""),
            L("This is time-sensitive for me, so I need to know what it means for me today.", "这件事对我比较紧急，我需要知道今天会有什么影响。"),
        ], "impact")

    if customer_turns == 2:
        return _varied_customer_move(state, [
            str(case.get("followup") or ""),
            L("I've already tried the basic steps on my side, so I don't want to start over again.", "我这边已经试过基本步骤了，不想再从头开始。"),
        ], "prior-context")

    # If customer-visible evidence conflicts with the company record, let the customer
    # naturally mention what they see once.  Do not keep re-asking for verification.
    if getattr(state, "conflicts", []) and not has_root:
        apparent = str(case.get("apparent") or "").strip()
        if apparent:
            return _varied_customer_move(state, [apparent], "visible-conflict")

    # The next useful customer move is to ask for the cause.  The resulting agent turn
    # receives the simulated diagnostic result immediately.
    if not has_root:
        root_questions = list(case.get("root_questions") or [])
        if _is_zh():
            root_questions = []
        return _varied_customer_move(state, root_questions + [
            L("What did you find, and can you fix the actual cause from your side?", "你查到具体原因了吗？如果可以，请直接从你们这边修复。"),
            L("So what is causing this, and can you take care of it from your side?", "所以具体是什么原因？你们这边可以直接处理吗？"),
        ], "diagnosis")

    # Once the cause is known, a real customer wants the fix.  This message authorizes
    # remediation; the simulator completes the backend action on this same turn so the
    # agent can respond with a resolved/not-resolved result instead of saying to wait.
    if has_root and authoritative_unresolved:
        fix_requests = list(case.get("fix_requests") or [])
        if _is_zh():
            fix_requests = []
        return _varied_customer_move(state, fix_requests + [
            L("Okay, that makes sense. Please go ahead and fix it.", "好的，这样就说得通了。请直接帮我修复。"),
            L("Understood. Please make that correction so we can get this resolved.", "明白了。请直接做这个更改，把问题解决掉。"),
            L("Thanks for explaining it. Yes, please take care of the fix on your side.", "谢谢你的解释。可以，请直接从你们这边处理好。"),
        ], "authorize-fix")

    return _varied_customer_move(state, [
        L("Okay, thanks. Please make sure it's fully taken care of.", "好的，谢谢。请确保这次彻底处理好。"),
        L("That makes sense. I just need this sorted out now.", "这样说得通。我现在只需要把问题处理好。"),
    ], "natural-default")

def _queue_manual_suggestion(suggestion: str) -> None:
    """Queue a prompt before the chat widget is instantiated on the next rerun.

    Streamlit forbids changing a widget's session-state value after that widget has
    already been created in the current run. Writing to a separate staging key avoids
    that API exception entirely.
    """
    st.session_state["manual_chat_prefill"] = suggestion


def render_start_here(domains: list[str]) -> None:
    st.markdown(L("## How to use JSpace Live", "## 如何使用 JSpace Live"))
    st.markdown(L(
        "Explore a generated service interaction or become the customer yourself. Customer messages appear immediately while the support agent is typing. The active workspace is deliberately small so you can see what evidence survives, what conflicts, and what changes the next response.",
        "你可以体验自动生成的客服场景，也可以自己扮演客户。客户消息会先立即显示，然后客服开始生成回复。活跃工作空间被刻意限制得很小，方便观察哪些证据被保留、哪些发生冲突，以及什么信息改变了下一步回复。",
    ))
    a, b = st.columns(2)
    with a:
        st.markdown(f'''<div class="j-card j-case"><div class="j-card-title">{L("MODE 01", "模式 01")}</div><div class="j-card-value">{L("Scenario Lab", "场景实验室")}</div><div class="j-card-meta">{L("Generate a controlled case. The customer appears one turn at a time, the support agent responds live, and the conversation only closes after confirmed resolution and a normal final check for other concerns.", "生成一个受控客服案例。客户逐轮出现，客服实时回复；只有在确认问题解决并正常询问是否还有其他问题后，对话才会结束。")}</div></div>''', unsafe_allow_html=True)
    with b:
        st.markdown(f'''<div class="j-card j-case"><div class="j-card-title">{L("MODE 02", "模式 02")}</div><div class="j-card-value">{L("Manual Multimodal AI", "手动多模态 AI")}</div><div class="j-card-meta">{L("You play the customer for as many turns as needed. Attach screenshots, voice clips or video in multimodal channels, or end the session whenever you are finished.", "你可以不限轮数地扮演客户。在多模态渠道中可上传截图、语音或视频，也可以随时结束会话。")}</div></div>''', unsafe_allow_html=True)

    st.markdown(L("### What the JSpace pipeline is doing", "### JSpace 流程在做什么"))
    nodes = [
        ("01", L("Signals", "信号"), L("Customer text, DeepSeek image evidence, Hy-ASR audio transcripts, YT-VITA video evidence, and company-system events.", "客户文字、DeepSeek 图像证据、Hy-ASR 音频转写、YT-VITA 视频证据以及公司系统事件。")),
        ("02", L("Concepts", "概念"), L("Signals become compact, traceable task-relevant concepts.", "把信号转换为紧凑、可追溯、与任务相关的概念。")),
        ("03", L("Conflict engine", "冲突引擎"), L("Contradictions remain explicit instead of being averaged away.", "矛盾信息会被显式保留，而不是被平均掉。")),
        ("04", "Top-K JSpace", L("Only a few concepts stay active; K=3–6 keeps the workspace genuinely compact.", "只保留少量活跃概念；K=3–6 让工作空间保持真正紧凑。")),
        ("05", L("Support response", "客服回复"), L("The agent reasons over the active state and recent conversation to move toward resolution.", "客服基于活跃状态和最近对话进行推理，并推动问题解决。")),
    ]
    node_html = "".join(f'<div class="j-node"><div class="j-node-num">NODE {n}</div><div class="j-node-name">{html.escape(name)}</div><div class="j-node-desc">{html.escape(desc)}</div></div>' for n, name, desc in nodes)
    st.markdown(f'<div class="j-node-grid">{node_html}</div>', unsafe_allow_html=True)

    st.markdown(L("### Key controls and signals", "### 关键控制与信号"))
    terms = [
        (L("JSpace capacity (K)", "JSpace 容量 (K)"), L("Maximum number of concepts allowed in the active workspace. Smaller K makes selection stricter; it does not directly control model latency.", "活跃工作空间允许保留的最大概念数量。K 越小，选择越严格；它不直接决定模型延迟。")),
        (L("Customer affect intensity", "客户情绪强度"), L("Scenario Lab can simulate nonverbal affect. In Manual mode, affect is inferred from typed/transcribed wording; Hy-ASR is transcription-only and does not classify vocal emotion.", "Scenario Lab 可以模拟非语言情绪。在 Manual 模式中，情绪只根据输入/转写文字推断；Hy-ASR 仅做转写，不判断声音情绪。")),
        (L("Satisfaction", "满意度"), L("A dynamic interaction-quality signal that rises with useful progress/resolution and falls when the exchange remains confusing or unhelpful.", "动态交互质量信号；当对话取得有效进展或解决问题时上升，持续混乱或无帮助时下降。")),
        (L("Priority", "优先级"), L("The ranking used to decide what survives Top-K: relevance, confidence, conflict importance and recency.", "决定哪些概念保留在 Top-K 中的排序信号：相关性、置信度、冲突重要性和时效性。")),
        (L("Evidence & provenance", "证据与来源"), L("Where each active concept came from — text, audio, image/video, backend systems, or derived reasoning.", "每个活跃概念来自哪里：文字、音频、图像/视频、后台系统或推导。")),
        (L("Researcher view", "研究者视图"), L("Hidden simulated company truth and provider diagnostics. Enable it explicitly in Settings.", "隐藏的模拟公司真值和提供方诊断信息。需要在设置中主动启用。")),
    ]
    cols = st.columns(3)
    for i, (name, desc) in enumerate(terms):
        with cols[i % 3]:
            st.markdown(f'<div class="j-domain"><strong>{html.escape(name)}</strong><span>{html.escape(desc)}</span></div>', unsafe_allow_html=True)

    st.markdown(L("### Customer-service domains", "### 客服领域"))
    cols = st.columns(3)
    for i, domain in enumerate(domains):
        with cols[i % 3]:
            st.markdown(f'<div class="j-domain"><strong>{html.escape(display_domain(domain))}</strong><span>{html.escape(domain_description(domain))}</span></div>', unsafe_allow_html=True)


def render_post_session_actions(state, profile: dict, domain: str, channel_label: str, *, mode: str) -> None:
    """Render post-session analysis and PDF export after a conversation ends."""
    if not state.session_ended:
        return
    st.markdown(L("#### Conversation wrap-up", "#### 对话结束总结"))
    analysis_key = f"conversation_analysis_{state.session_id}"
    provider_key = f"conversation_analysis_provider_{state.session_id}"
    a1, a2 = st.columns(2)
    with a1:
        if st.button(L("Analyze conversation", "分析对话"), key=f"analyze_{state.session_id}", use_container_width=True):
            with st.spinner(L("Analyzing conversation…", "正在分析对话…")):
                cfg = _ai_runtime()
                analysis, analysis_provider = analyze_conversation_summary(
                    transcript=state.transcript, domain=domain, channel=CHANNELS[channel_label]["slug"],
                    profile=profile, satisfaction=state.customer_satisfaction, conflicts=len(state.conflicts),
                    api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL,
                    timeout_s=cfg["timeout_ms"] / 1000.0, language=_language_prompt_name(),
                )
                st.session_state[analysis_key] = analysis
                st.session_state[provider_key] = analysis_provider
                st.rerun()
    analysis_text = st.session_state.get(analysis_key, "")
    # v1.4.6: build the website download with the canonical in-app plain-text
    # transcript renderer. This deliberately bypasses backend.conversation_export
    # so a stale Streamlit module can never resurrect the legacy chat-card PDF.
    pdf_bytes = build_website_plain_transcript_pdf(
        transcript=state.transcript, profile=profile, domain=display_domain(domain), channel=display_channel(channel_label),
        session_id=state.session_id, satisfaction=state.customer_satisfaction, phase=state.session_phase,
        language=_language_prompt_name(), analysis=analysis_text or None,
    )
    with a2:
        st.download_button(
            L("Save conversation as PDF", "将对话保存为 PDF"), data=pdf_bytes,
            file_name=f"jspace_{mode}_{state.session_id}_transcript.pdf", mime="application/pdf",
            # Version the widget key so Streamlit/browser state cannot retain an older
            # download payload after a hot redeploy of the PDF implementation.
            key=f"download_pdf_plain_v146_{state.session_id}", use_container_width=True,
        )
    if analysis_text:
        st.markdown(analysis_text)
        provider = st.session_state.get(provider_key, "")
        if provider:
            st.caption(L("Analysis provider · ", "分析来源 · ") + provider)


def process_scenario_turn(scenario, state, step_index: int, channel_label: str, conversation_slot) -> bool:
    epoch = int(st.session_state.get("generation_epoch", 0))
    step = scenario.steps[step_index]
    apply_scenario_customer_step(scenario, state, step_index)
    # Customer message is emitted immediately before the network request starts.
    render_conversation(state.transcript, channel_label, slot=conversation_slot)
    render_conversation(state.transcript, channel_label, typing=True, slot=conversation_slot)
    reply, provider = stream_agent_reply(
        state, scenario.customer_profile, scenario.domain, channel_label, conversation_slot, epoch=epoch
    )
    if reply is None or int(st.session_state.get("generation_epoch", 0)) != epoch:
        return False
    append_agent_reply(state, reply, provider, step_label=step.label)
    update_customer_relationship(scenario.customer_profile, state, reply, provider)
    render_conversation(state.transcript, channel_label, slot=conversation_slot)
    return True


def process_manual_turn(state, profile, domain: str, channel_label: str, prompt: str, media_files, conversation_slot, *, video_url: str = "") -> bool:
    epoch = int(st.session_state.get("generation_epoch", 0))
    media, media_display = read_uploaded_media(media_files)
    video_url = (video_url or "").strip()
    if video_url:
        media.append({"name": "linked-video", "mime_type": "video/mp4", "url": video_url})
        media_display.append({"name": "linked video URL", "mime_type": "video/mp4"})
    apply_manual_customer_message(
        state, prompt, attachments=media_display, affect_source="text"
    )
    # Show the customer's message immediately. Media analysis only runs when the user attached media.
    render_conversation(state.transcript, channel_label, slot=conversation_slot)
    render_conversation(state.transcript, channel_label, typing=True, slot=conversation_slot)
    if media:
        cfg = _ai_runtime()
        media_kwargs = dict(
            api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, audio_model=TOKENHUB_AUDIO_MODEL,
            video_model=TOKENHUB_VIDEO_MODEL, base_url=TOKENHUB_BASE_URL, domain=domain,
            timeout_s=cfg["timeout_ms"] / 1000.0,
        )
        if _accepts_kwarg(analyze_media_for_jspace, "language"):
            media_kwargs["language"] = _language_prompt_name()
        media_concepts = analyze_media_for_jspace(media, **media_kwargs)
        if int(st.session_state.get("generation_epoch", 0)) != epoch:
            return False
        if media_concepts:
            merge_concepts(state.concepts, media_concepts)
            refresh_state(state)
    reply, provider = stream_agent_reply(
        state, profile, domain, channel_label, conversation_slot, media=media, epoch=epoch
    )
    if reply is None or int(st.session_state.get("generation_epoch", 0)) != epoch:
        return False
    append_agent_reply(state, reply, provider)
    update_customer_relationship(profile, state, reply, provider)
    if float(profile.get("patience", 0)) < 0:
        state.session_phase = "ended"
        state.session_ended = True
    render_conversation(state.transcript, channel_label, slot=conversation_slot)
    return True


domains = list_domains()
start_tab, scenario_tab, manual_tab = st.tabs(
    [L("◎ Start Here", "◎ 开始"), L("✦ Scenario Lab", "✦ 场景实验室"), L("◈ Manual Multimodal AI", "◈ 手动多模态 AI")],
    on_change=_on_main_tab_change,
    key="main_tabs",
)

if start_tab.open:
    with start_tab:
        render_start_here(domains)

if scenario_tab.open:
    with scenario_tab:
        st.markdown(L("## Scenario Lab", "## 场景实验室"))
        control1, control2, control3 = st.columns([1.25, 1, 1])
        with control1:
            scenario_domain = st.selectbox(
                L("Domain", "领域"), ["random"] + domains, key="scenario_domain",
                format_func=lambda d: L("Random", "随机") if d == "random" else display_domain(d),
            )
            if scenario_domain != "random":
                st.caption(domain_description(scenario_domain))
        with control2:
            channel_label = st.selectbox(
                L("Conversation channel", "对话渠道"), SCENARIO_CHANNELS, index=0, key="scenario_channel", format_func=display_channel,
            )
            st.caption(channel_hint(channel_label))
        with control3:
            scenario_k = st.slider(L("JSpace capacity K", "JSpace 容量 K"), 3, 6, 4, key="scenario_k")
            seed_text = st.text_input(L("Optional seed", "可选随机种子"), placeholder=L("blank = new case", "留空 = 新案例"), key="scenario_seed")
            seed = int(seed_text) if seed_text.strip().isdigit() else None

        if st.button(L("Generate scenario", "生成场景"), type="primary", use_container_width=True, key="generate_scenario"):
            _bump_generation_epoch()
            epoch = int(st.session_state.generation_epoch)
            with st.spinner(L("Building a customer case…", "正在生成客户案例…")):
                scenario = generate_scenario(ScenarioControls(domain=scenario_domain, seed=seed))
                scenario_provider = L("Curated scenario · instant", "预设场景 · 即时")
                if AI_CONNECTED and (_setting("scenario_ai") or _is_zh()):
                    cfg = _ai_runtime()
                    rewrite_kwargs = dict(
                        api_key=TOKENHUB_API_KEY,
                        model=TOKENHUB_MODEL,
                        base_url=TOKENHUB_BASE_URL,
                        channel=CHANNELS[channel_label]["slug"],
                        timeout_s=cfg["scenario_timeout_ms"] / 1000.0,
                    )
                    if _accepts_kwarg(enhance_scenario_with_deepseek, "language"):
                        rewrite_kwargs["language"] = _language_prompt_name()
                    scenario, scenario_provider = enhance_scenario_with_deepseek(scenario, **rewrite_kwargs)
                scenario = prepare_scenario_for_channel(scenario, channel_label)
                scenario = localize_generated_scenario(scenario)
            if int(st.session_state.get("generation_epoch", 0)) == epoch:
                st.session_state.live_scenario = scenario
                st.session_state.live_state = new_scenario_state(scenario, capacity_k=scenario_k)
                st.session_state.live_next_step = 0
                st.session_state.live_started = False
                st.session_state.live_channel = channel_label
                st.session_state.live_scenario_provider = scenario_provider
                st.session_state.live_user_ended = False
                st.session_state.scenario_scroll_pending = True
                st.rerun()

        scenario = st.session_state.get("live_scenario")
        state = st.session_state.get("live_state")
        next_step = st.session_state.get("live_next_step", 0)
        live_channel = st.session_state.get("live_channel", channel_label)

        if not scenario or not state:
            st.info(L("Generate a scenario first. The case brief will appear here, then you can start the live conversation.", "请先生成场景。这里会先显示案例摘要，然后你可以开始实时对话。"))
        else:
            st.markdown('<div id="scenario-live-anchor"></div>', unsafe_allow_html=True)
            if st.session_state.pop("scenario_scroll_pending", False) and _setting("auto_scroll"):
                _scroll_to("scenario-live-anchor")

            st.markdown(
                f'''<div class="j-card j-case"><div class="j-card-title">{L("CASE BRIEF", "案例摘要")} · {html.escape(display_domain(scenario.domain))}</div><div class="j-card-value">{html.escape(scenario.title)}</div><div class="j-card-meta">{html.escape(scenario.problem_summary or scenario.steps[0].customer_turn.text)} · {html.escape(channel_hint(live_channel))}</div></div>''',
                unsafe_allow_html=True,
            )
            render_profile(scenario.customer_profile, state)

            chat_col, workspace_col = st.columns([1.14, .86], gap="large")
            with chat_col:
                conversation_slot = st.empty()
                render_conversation(state.transcript, live_channel, slot=conversation_slot)
                st.caption(L("Scrollable conversation · newest turn stays in view.", "对话可滚动 · 自动跟随最新一轮消息。"))
                if not st.session_state.get("live_started", False) and not state.session_ended:
                    start_c, end_c = st.columns([3, 1])
                    if start_c.button(L("Start conversation ▶", "开始对话 ▶"), type="primary", use_container_width=True, key="start_live"):
                        st.session_state.live_started = True
                        ok = process_scenario_turn(scenario, state, 0, live_channel, conversation_slot)
                        if ok:
                            st.session_state.live_next_step = 1
                            st.session_state.live_state = state
                        st.rerun()
                    if end_c.button(L("End session", "结束会话"), use_container_width=True, key="end_scenario_before_start"):
                        _bump_generation_epoch()
                        end_scenario_session(state)
                        st.session_state.live_user_ended = True
                        st.session_state.live_state = state
                        st.rerun()
                elif not state.session_ended and next_step < len(scenario.steps):
                    next_c, end_c = st.columns([3, 1])
                    if next_c.button(L("Continue conversation →", "继续对话 →"), type="primary", use_container_width=True, key=f"continue_scenario_{next_step}"):
                        ok = process_scenario_turn(scenario, state, next_step, live_channel, conversation_slot)
                        if ok:
                            st.session_state.live_next_step = next_step + 1
                            st.session_state.live_state = state
                        st.rerun()
                    if end_c.button(L("End session", "结束会话"), use_container_width=True, key=f"end_scenario_{next_step}"):
                        _bump_generation_epoch()
                        end_scenario_session(state)
                        st.session_state.live_user_ended = True
                        st.session_state.live_state = state
                        st.rerun()
                elif state.session_ended:
                    if st.session_state.get("live_user_ended", False):
                        st.info(L("Practice session ended by the user. The app does not claim that the unresolved case was resolved.", "练习会话已由用户结束；应用不会把仍未解决的问题错误标记为已解决。"))
                    else:
                        st.success(L("Conversation closed naturally after confirmed resolution and the customer's final check-in.", "问题确认解决并完成最后确认后，对话已自然结束。"))
                    render_post_session_actions(state, scenario.customer_profile, scenario.domain, live_channel, mode="scenario")

            with workspace_col:
                render_workspace(state, show_coaching=True)

            if _setting("researcher_view"):
                with st.expander(L("Researcher view · scenario ground truth", "研究者视图 · 场景真值"), expanded=False):
                    st.write(L("**Domain:**", "**领域：**"), display_domain(scenario.domain))
                    st.write(L("**Problem summary:**", "**问题摘要：**"), scenario.problem_summary)
                    st.write(L("**Random conflict present:**", "**是否存在随机冲突：**"), scenario.expected_conflict)
                    st.write(L("**Hidden ground truth:**", "**隐藏真值：**"), scenario.hidden_ground_truth)
                    st.write(L("**Scenario language source:**", "**场景语言来源：**"), st.session_state.get("live_scenario_provider", L("Curated scenario", "预设场景")))
                    st.write(L("**Seed:**", "**随机种子：**"), scenario.seed)
                    providers = [r.get("provider") for r in state.transcript if r.get("role") == "agent"]
                    st.write(L("**Agent providers:**", "**客服模型来源：**"), providers)

if manual_tab.open:
    with manual_tab:
        st.markdown(L("## Manual Multimodal AI", "## 手动多模态 AI"))
        m1, m2, m3 = st.columns([1.25, 1, 1])
        with m1:
            manual_domain = st.selectbox(L("Domain", "领域"), domains, key="manual_domain", format_func=display_domain)
            st.caption(domain_description(manual_domain))
        with m2:
            manual_channel = st.selectbox(L("Channel / input mode", "渠道 / 输入模式"), list(MANUAL_MODE_CONFIG), index=4, key="manual_channel", format_func=display_channel)
            st.caption(channel_hint(manual_channel))
        with m3:
            manual_k = st.slider(L("JSpace capacity K", "JSpace 容量 K"), 3, 6, 4, key="manual_k")
            start_manual = st.button(L("Start / reset session", "开始 / 重置会话"), type="primary", use_container_width=True)

        if start_manual:
            _bump_generation_epoch()
            profile, backend_events = generate_manual_context(manual_domain)
            st.session_state.manual_state_v06 = new_manual_state(capacity_k=manual_k, backend_events=backend_events, profile=profile)
            st.session_state.manual_profile_v06 = profile
            st.session_state.manual_domain_v06 = manual_domain
            st.session_state.manual_channel_v06 = manual_channel
            st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
            st.session_state.manual_chat_input = ""
            st.rerun()

        manual_state = st.session_state.get("manual_state_v06")
        manual_profile = st.session_state.get("manual_profile_v06")
        active_manual_domain = st.session_state.get("manual_domain_v06", manual_domain)
        active_manual_channel = st.session_state.get("manual_channel_v06", manual_channel)

        if manual_state and manual_profile:
            st.markdown(
                f'''<div class="j-card j-case"><div class="j-card-title">{L("PRACTICE CASE", "练习案例")} · {html.escape(display_domain(active_manual_domain))}</div><div class="j-card-value">{L("You are the customer. Continue for as many turns as needed.", "你扮演客户，可以不限轮数继续对话。")}</div><div class="j-card-meta">{L("The company record is simulated automatically. In multimodal channels you can attach media that may support or contradict what you say.", "公司系统记录由应用自动模拟。在多模态渠道中，你可以添加支持或反驳自己说法的媒体证据。")}</div></div>''',
                unsafe_allow_html=True,
            )
            render_profile(manual_profile, manual_state)

            chat_col, workspace_col = st.columns([1.18, .82], gap="large")
            with chat_col:
                conversation_slot = st.empty()
                render_conversation(manual_state.transcript, active_manual_channel, slot=conversation_slot)

                if not manual_state.session_ended:
                    mode_cfg = manual_mode_config(active_manual_channel)
                    suggestion = suggested_customer_prompt(active_manual_domain, manual_state) if mode_cfg["show_suggestion"] else ""
                    if st.session_state.pop("manual_input_reset_pending", False):
                        st.session_state["manual_chat_input"] = ""
                    queued_prefill = st.session_state.pop("manual_chat_prefill", None)
                    if queued_prefill is not None:
                        st.session_state["manual_chat_input"] = queued_prefill
                    st.session_state.setdefault("manual_chat_input", "")

                    prompt = ""
                    send = False
                    if mode_cfg["allow_text"]:
                        with st.container(key="manual_composer"):
                            compose_col, suggest_col = st.columns([4.2, 1.3], gap="small", vertical_alignment="bottom")
                            with compose_col:
                                with st.form("manual_chat_form", clear_on_submit=False):
                                    input_col, send_col = st.columns([5.2, 1.15], gap="small", vertical_alignment="bottom")
                                    with input_col:
                                        prompt = st.text_input(
                                            L("Customer message", "客户消息"),
                                            key="manual_chat_input",
                                            placeholder=L(*mode_cfg["placeholder"]),
                                            label_visibility="collapsed",
                                        )
                                    with send_col:
                                        send = st.form_submit_button(L("Send", "发送"), type="primary", use_container_width=True)
                            with suggest_col:
                                if mode_cfg["show_suggestion"]:
                                    st.markdown(
                                        f'<div class="j-suggest-mini"><strong>{L("Suggested prompt", "建议提示")}</strong><br>{html.escape(suggestion)}</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.button(
                                        L("Use prompt", "填入输入框"),
                                        key=f"use_manual_suggestion_{sum(1 for r in manual_state.transcript if r.get('role') == 'agent')}",
                                        use_container_width=True,
                                        help=L("Fill the chat box immediately; then press Enter or Send.", "立即填入聊天框，然后按 Enter 或点击发送。"),
                                        on_click=_queue_manual_suggestion,
                                        args=(suggestion,),
                                    )
                    else:
                        st.markdown(
                            f'<div class="j-suggest-mini"><strong>{L("Selected input mode", "当前输入模式")}</strong><br>{html.escape(channel_hint(active_manual_channel))}</div>',
                            unsafe_allow_html=True,
                        )

                    media_files = []
                    video_url = ""
                    allowed_types = mode_cfg["file_types"]
                    if allowed_types or mode_cfg["allow_video_url"]:
                        evidence_title = {
                            "Image Upload": L("Image evidence", "图片证据"),
                            "Audio Upload": L("Audio evidence", "音频证据"),
                            "Video Upload": L("Video evidence", "视频证据"),
                            "Multimodal Mix": L("Multimodal evidence", "多模态证据"),
                        }.get(active_manual_channel, L("Evidence for this turn", "本轮证据"))
                        with st.expander(evidence_title, expanded=True):
                            st.caption(channel_hint(active_manual_channel))
                            if allowed_types:
                                media_files = st.file_uploader(
                                    L("Attach evidence for this turn", "为本轮添加证据"),
                                    type=allowed_types, accept_multiple_files=True,
                                    key=f"manual_media_{st.session_state.get('manual_media_key', 0)}",
                                    help=L("Only files supported by the selected mode are accepted in this uploader.", "上传器只接受当前模式支持的文件类型。"),
                                )
                                if media_files:
                                    st.caption(L("Attached: ", "已添加：") + ", ".join(f.name for f in media_files))
                            if mode_cfg["allow_video_url"]:
                                video_url = st.text_input(
                                    L("Public video URL", "公开视频 URL"),
                                    key=f"manual_video_url_{st.session_state.get('manual_media_key', 0)}",
                                    placeholder="https://.../clip.mp4",
                                    help=L("A public URL is the most reliable YT-VITA path; local uploads use a best-effort data URL.", "公开 URL 是 YT-VITA 最可靠的方式；本地上传会尝试使用 data URL。"),
                                )
                        if not mode_cfg["allow_text"]:
                            send = st.button(
                                L("Analyze & send evidence", "分析并发送证据"),
                                type="primary", use_container_width=True,
                                key=f"manual_media_submit_{active_manual_channel}",
                            )

                    end_now = st.button(L("End session", "结束会话"), use_container_width=True, key="manual_end_session")


                    if send and ((prompt or "").strip() or media_files or (video_url or "").strip()):
                        customer_text = (prompt or "").strip() or L("[Customer attached media evidence for this turn.]", "[客户为本轮添加了媒体证据。]")
                        ok = process_manual_turn(
                            manual_state, manual_profile, active_manual_domain, active_manual_channel,
                            customer_text, media_files, conversation_slot, video_url=video_url,
                        )
                        if ok:
                            st.session_state.manual_state_v06 = manual_state
                            st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
                            st.session_state.manual_input_reset_pending = True
                        st.rerun()
                    if end_now:
                        _bump_generation_epoch()
                        end_manual_session(manual_state)
                        st.session_state.manual_state_v06 = manual_state
                        st.rerun()
                else:
                    if float(manual_profile.get("patience", 0)) < 0:
                        st.warning(L("The customer ended the conversation because their patience was exhausted.", "客户耐心已经耗尽，因此主动结束了本次对话。"))
                    else:
                        st.success(L("Session ended. Start/reset a session whenever you want to practice another conversation.", "会话已结束。需要练习新的对话时，可以随时开始或重置会话。"))
                    render_post_session_actions(manual_state, manual_profile, active_manual_domain, active_manual_channel, mode="manual")
            with workspace_col:
                render_workspace(manual_state, show_coaching=False)

            if _setting("researcher_view"):
                with st.expander(L("Researcher view · simulated company context", "研究者视图 · 模拟公司上下文"), expanded=False):
                    st.write(L("**Domain:**", "**领域：**"), display_domain(active_manual_domain))
                    st.write(L("**Channel:**", "**渠道：**"), display_channel(active_manual_channel))
                    st.write(L("**Company-system events:**", "**公司系统事件：**"), manual_state.backend_history)
                    st.write(L("**AI connected:**", "**AI 是否连接：**"), AI_CONNECTED)
                    st.write(L("**Provider:**", "**提供方：**"), "Tencent TokenHub")
                    st.write(L("**Text + image model:**", "**文字 + 图像模型：**"), TOKENHUB_MODEL)
                    st.write(L("**Audio transcription:**", "**音频转写：**"), TOKENHUB_AUDIO_MODEL)
                    st.write(L("**Video understanding:**", "**视频理解：**"), TOKENHUB_VIDEO_MODEL)
                    st.write(L("**Agent providers:**", "**客服模型来源：**"), [r.get("provider") for r in manual_state.transcript if r.get("role") == "agent"])
        else:
            st.info(L("Choose a domain/channel and start a session. Multimodal Mix gives the richest demonstration of conflicting evidence across modalities.", "选择领域和渠道并开始会话。多模态混合模式最适合展示不同模态之间的证据冲突。"))

st.markdown("---")
st.caption(L("JSpace Live · capacity-limited, conflict-aware multimodal customer-service research experience", "JSpace Live · 容量受限、冲突感知的多模态客服研究体验"))
