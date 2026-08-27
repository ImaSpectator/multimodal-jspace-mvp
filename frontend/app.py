from __future__ import annotations

import html
import inspect
import os
import sys
import urllib.parse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.jspace.ai_provider import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_VIDEO_MODEL,
    analyze_media_for_jspace,
    enhance_scenario_with_deepseek,
    generate_support_reply,
    probe_deepseek,
    stream_support_reply,
)
from backend.jspace.engine import merge_concepts, refresh_state  # noqa: E402
from backend.jspace.scenario_generator import generate_manual_context, generate_scenario, list_domains  # noqa: E402
from backend.jspace.schemas import ImageObservation, ScenarioControls  # noqa: E402
from backend.jspace.simulator import (  # noqa: E402
    append_agent_reply,
    apply_manual_customer_message,
    apply_scenario_customer_step,
    end_manual_session,
    end_scenario_session,
    new_manual_state,
    new_scenario_state,
)

APP_VERSION = "0.9.0-manual-ux-language"

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

/* Compact top-right utility toolbar. Use icon-only buttons so labels never wrap vertically. */
.st-key-utility_toolbar { margin-top:-.08rem; margin-bottom:.48rem; }
.st-key-utility_toolbar [data-testid="stHorizontalBlock"] { align-items:center!important; gap:.38rem!important; }
.st-key-utility_toolbar [data-testid="column"]:first-child { flex:1 1 auto!important; min-width:0!important; }
.st-key-utility_toolbar [data-testid="column"]:nth-child(2) {
  flex:0 0 3.65rem!important; width:3.65rem!important; min-width:3.65rem!important; max-width:3.65rem!important;
}
.st-key-utility_toolbar [data-testid="column"]:nth-child(n+3) {
  flex:0 0 2.70rem!important; width:2.70rem!important; min-width:2.70rem!important; max-width:2.70rem!important;
}
.st-key-utility_toolbar .stButton { margin:0!important; width:100%!important; }
.st-key-utility_toolbar .stButton > button {
  position:relative!important; width:100%!important; height:2.70rem!important; min-height:2.70rem!important; padding:0!important;
  border-radius:13px!important; border:1px solid rgba(93,245,255,.24)!important;
  background:linear-gradient(145deg,rgba(14,29,49,.94),rgba(8,17,31,.94))!important;
  color:#DFF8FF!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 7px 20px rgba(0,0,0,.15)!important;
  transition:.16s ease!important;
}
.st-key-utility_toolbar .stButton > button:hover {
  transform:translateY(-1px); border-color:rgba(93,245,255,.55)!important;
  background:linear-gradient(145deg,rgba(18,43,65,.98),rgba(13,25,45,.98))!important;
  box-shadow:0 8px 24px rgba(58,177,220,.12),inset 0 1px 0 rgba(255,255,255,.06)!important;
}
.st-key-utility_toolbar .stButton > button p {
  margin:0!important; font-size:.78rem!important; font-weight:800!important; letter-spacing:.01em!important;
  white-space:nowrap!important; line-height:1!important;
}
.st-key-utility_toolbar .stButton > button > div {
  position:absolute!important; inset:0!important; width:100%!important; height:100%!important;
  display:flex!important; align-items:center!important; justify-content:center!important; gap:0!important; margin:0!important; padding:0!important;
}
.st-key-top_help .stButton > button p,
.st-key-top_share .stButton > button p,
.st-key-top_reset .stButton > button p,
.st-key-top_settings .stButton > button p { display:none!important; }
.st-key-utility_toolbar .stButton > button [data-testid="stIconMaterial"] {
  position:static!important; transform:none!important; display:block!important;
  font-size:1.16rem!important; width:1.16rem!important; height:1.16rem!important; line-height:1.16rem!important;
  margin:0!important; padding:0!important; text-align:center!important;
}
.st-key-top_language .stButton > button > div { display:flex!important; align-items:center!important; justify-content:center!important; }
.st-key-top_language .stButton > button p { display:block!important; margin:0!important; width:auto!important; text-align:center!important; }
@media(max-width:700px){
  .st-key-utility_toolbar [data-testid="column"]:nth-child(2){flex-basis:3.35rem!important;width:3.35rem!important;min-width:3.35rem!important;max-width:3.35rem!important;}
  .st-key-utility_toolbar [data-testid="column"]:nth-child(n+3){flex-basis:2.45rem!important;width:2.45rem!important;min-width:2.45rem!important;max-width:2.45rem!important;}
  .st-key-utility_toolbar .stButton>button{height:2.45rem!important;min-height:2.45rem!important;}
}
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


def _init_preferences() -> None:
    defaults = {
        "settings_speed": "Fast",
        "settings_reply": "Concise",
        "settings_scenario_ai": True,
        "settings_auto_scroll": True,
        "ui_language": "English",
        "generation_epoch": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _ai_runtime() -> dict:
    speed = st.session_state.get("settings_speed", "Fast")
    reply = st.session_state.get("settings_reply", "Concise")
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


@st.dialog("Share" if not _is_zh() else "分享", width="small")
def _share_dialog() -> None:
    st.markdown(L("**Email this experience**", "**通过邮件分享**"))
    share_url = st.text_input(L("Public app link", "公开应用链接"), value=PUBLIC_APP_URL, placeholder="https://your-app.streamlit.app", key="share_url")
    recipient = st.text_input(L("Recipient email", "收件人邮箱"), placeholder="name@example.com", key="share_email")
    if share_url and recipient and "@" in recipient:
        subject = urllib.parse.quote("Try JSpace Live" if not _is_zh() else "体验 JSpace Live")
        body_text = f"I wanted to share this JSpace customer-service experience with you:\n\n{share_url}" if not _is_zh() else f"想和你分享这个 JSpace 客服体验：\n\n{share_url}"
        body = urllib.parse.quote(body_text)
        st.link_button(L("Email link", "打开邮件"), f"mailto:{urllib.parse.quote(recipient)}?subject={subject}&body={body}", width="stretch")
        st.caption(L("This opens your email app; you press Send there.", "这会打开你的邮件应用，最后由你点击发送。"))
    elif recipient and "@" not in recipient:
        st.caption(L("Enter a valid email address.", "请输入有效的邮箱地址。"))


@st.dialog("Settings" if not _is_zh() else "设置", width="small")
def _settings_dialog() -> None:
    st.markdown(L("**Conversation settings**", "**对话设置**"))
    st.selectbox(
        L("AI response profile", "AI 响应速度"), ["Fast", "Balanced"], key="settings_speed",
        help=L("Fast uses a 12-second DeepSeek attempt cap and 4 recent messages. Balanced uses a 20-second cap and 6 recent messages.", "Fast：单次 DeepSeek 最长 12 秒、保留最近 4 条消息；Balanced：最长 20 秒、保留最近 6 条消息。"),
    )
    st.selectbox(L("Agent reply length", "客服回复长度"), ["Concise", "Standard"], key="settings_reply")
    st.toggle(L("Use DeepSeek to vary scenario wording", "使用 DeepSeek 改写场景措辞"), key="settings_scenario_ai", help=L("Turn this off for near-instant curated scenario generation.", "关闭后会直接使用预设场景，生成速度更快。"))
    st.toggle(L("Auto-scroll conversations", "对话自动滚动到底部"), key="settings_auto_scroll")
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
    spacer, lang_col, u1, u2, u3, u4 = st.columns([1, .085, .058, .058, .058, .058], gap="small", vertical_alignment="center")
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
        if st.button(" ", icon=":material/share:", help=L("Share", "分享"), key="top_share", width="stretch"):
            _share_dialog()
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
    c1.caption(f"{L('Patience', '耐心度')} · {profile.get('patience', 0)}/100")
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
            L("Concept", "概念"): c.name.replace("_", " ").title(),
            L("Value", "值"): c.value,
            L("Status", "状态"): STATUS_ZH.get(c.status, c.status) if _is_zh() else c.status,
            L("Sources", "来源"): ", ".join(SOURCE_ZH.get(src, src) if _is_zh() else src for src in c.sources),
            L("Priority", "优先级"): round(c.score, 2),
            L("Confidence", "置信度"): round(c.confidence, 2),
        })
    return pd.DataFrame(rows)


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
                f'<div class="j-card j-conflict"><div class="j-card-title">{html.escape((conflict.severity.upper()+" SIGNAL CONFLICT") if not _is_zh() else ("信号冲突 · "+conflict.severity))}</div><div class="j-card-value">{html.escape(conflict.description)}</div></div>',
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
                f'''<div class="j-card j-concept {html.escape(c.status)}"><div class="j-card-title">{html.escape(c.name.replace('_',' ').title())}</div><div class="j-card-value">{html.escape(str(c.value))}</div><div class="j-card-meta">{html.escape(sources)} · {L('priority', '优先级')} {c.score:.2f} · {html.escape(STATUS_ZH.get(c.status, c.status) if _is_zh() else c.status)}</div></div>''',
                unsafe_allow_html=True,
            )

    status_concepts = status_workspace_concepts(state)
    if status_concepts:
        with st.expander(L("Resolution/status context", "解决状态上下文"), expanded=False):
            st.caption(L("Status remains available for conflict reasoning but no longer dominates the primary concept display.", "状态信息仍会参与冲突推理，但不会再占据主要概念展示的顶部。"))
            for c in status_concepts:
                sources = " · ".join((SOURCE_ZH.get(src, src) if _is_zh() else src.title()) for src in c.sources)
                st.markdown(
                    f'''<div class="j-card j-concept {html.escape(c.status)}"><div class="j-card-title">{html.escape(c.name.replace('_',' ').title())}</div><div class="j-card-value">{html.escape(str(c.value))}</div><div class="j-card-meta">{html.escape(sources)} · {L('priority', '优先级')} {c.score:.2f}</div></div>''',
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
    if st.session_state.get("settings_auto_scroll", True) and transcript:
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


def suggested_customer_prompt(domain: str, state) -> str:
    if not state.transcript:
        return (CUSTOMER_STARTERS_ZH if _is_zh() else CUSTOMER_STARTERS).get(
            domain, L("I need help with an issue on my account. Can you check the current system status?", "我的账户有一个问题。你能帮我检查当前系统状态吗？")
        )
    if state.session_ended:
        return ""

    last_agent = ""
    for row in reversed(state.transcript):
        if row.get("role") == "agent":
            last_agent = str(row.get("text") or "")
            break
    low = last_agent.lower()

    if state.conflicts:
        return L(
            "What I'm seeing still doesn't match what you're telling me. Which system is authoritative, and what exactly is causing the mismatch?",
            "我看到的信息还是和你说的不一致。到底哪个系统才是权威记录？造成不一致的具体原因是什么？",
        )
    if state.session_phase == "resolved":
        return L(
            "Thanks. Can you confirm there isn't anything else I need to do on my side before we wrap up?",
            "谢谢。结束之前，你能确认我这边已经不需要再做其他事情了吗？",
        )
    if any(k in low for k in ["confirmation number", "reservation number", "order number", "booking reference", "ticket number"]):
        return L(
            "Sure — the reference is on hand. Before I share it, can you tell me exactly what you'll verify once you pull it up?",
            "可以，我这边有编号。不过在我发给你之前，你能先告诉我你调出来之后会具体核实什么吗？",
        )
    if any(k in low for k in ["24", "48", "business day", "within", "refund", "timeframe", "eta"]):
        return L(
            "Thanks. What exact timeline should I expect, and what should I do if that deadline passes without an update?",
            "好的。那我应该预期一个怎样的具体时间线？如果过了这个时间还没有更新，我应该怎么做？",
        )
    if any(k in low for k in ["verify", "checking", "investigat", "look into", "review"]):
        return L(
            "Can you tell me what you've already verified so far and what the next concrete step is?",
            "你能告诉我目前已经核实了哪些信息，以及下一步具体会做什么吗？",
        )
    if any(k in low for k in ["blocker", "cause", "root cause", "preventing", "holding"]):
        return L(
            "Can you explain the blocker in plain language and what action will remove it?",
            "你能用直白一点的话解释这个阻塞点是什么，以及要采取什么动作才能解除它吗？",
        )
    turn_count = sum(1 for row in state.transcript if row.get("role") == "agent")
    variants = [
        L("What is the next action you'll take on your side to move this forward?", "你这边下一步会采取什么动作来推进处理？"),
        L("Can you be specific about what changed in the system versus what still hasn't changed yet?", "你能具体说说系统里已经发生了什么变化、还有哪些地方还没变吗？"),
        L("Before we move on, what is the most important thing I should understand about the case right now?", "在继续之前，目前这个问题里我最应该先理解清楚的关键信息是什么？"),
    ]
    return variants[turn_count % len(variants)]


def _use_manual_suggestion(suggestion: str) -> None:
    """Prefill the manual chat widget on the next rerun."""
    st.session_state["manual_chat_input"] = suggestion


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
        (L("Researcher view", "研究者视图"), L("Hidden simulated company truth and provider diagnostics. Closed by default because a real customer would not see it.", "隐藏的模拟公司真值和提供方诊断信息。默认关闭，因为真实客户不会看到这些信息。")),
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
        media_concepts = analyze_media_for_jspace(
            media, api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, audio_model=TOKENHUB_AUDIO_MODEL,
            video_model=TOKENHUB_VIDEO_MODEL, base_url=TOKENHUB_BASE_URL, domain=domain,
            timeout_s=cfg["timeout_ms"] / 1000.0,
        )
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
                if AI_CONNECTED and (st.session_state.get("settings_scenario_ai", True) or _is_zh()):
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
            if st.session_state.pop("scenario_scroll_pending", False) and st.session_state.get("settings_auto_scroll", True):
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
                if st.session_state.get("live_started", False) and not state.session_ended:
                    completed_turns = min(next_step, len(scenario.steps))
                    st.caption(L(f"Scenario progress · {completed_turns}/{len(scenario.steps)} customer turns revealed", f"场景进度 · 已展示 {completed_turns}/{len(scenario.steps)} 轮客户消息"))

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

            with workspace_col:
                render_workspace(state, show_coaching=True)

            with st.expander(L("Researcher view · scenario ground truth", "研究者视图 · 场景真值"), expanded=False):
                st.write(L("**Domain:**", "**领域：**"), display_domain(scenario.domain))
                st.write(L("**Problem summary:**", "**问题摘要：**"), scenario.problem_summary)
                st.write(L("**Random conflict present:**", "**是否存在随机冲突：**"), scenario.expected_conflict)
                st.write(L("**Hidden ground truth:**", "**隐藏真值：**"), scenario.hidden_ground_truth)
                st.write(L("**Scenario language source:**", "**场景语言来源：**"), st.session_state.get("live_scenario_provider", L("Curated scenario", "预设场景")))
                st.write(L("**Seed:**", "**随机种子：**"), scenario.seed)
                st.write(L("**Planned customer turns:**", "**计划客户轮数：**"), len(scenario.steps))
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
                                    if st.button(
                                        L("Use prompt", "填入输入框"),
                                        key="use_manual_suggestion",
                                        use_container_width=True,
                                        help=L("Fill the chat box immediately; then press Enter or Send.", "立即填入聊天框，然后按 Enter 或点击发送。"),
                                    ):
                                        st.session_state["manual_chat_input"] = suggestion
                                        st.rerun()
                    else:
                        st.markdown(
                            f'<div class="j-suggest-mini"><strong>{L("Selected input mode", "当前输入模式")}</strong><br>{html.escape(channel_hint(active_manual_channel))}</div>',
                            unsafe_allow_html=True,
                        )

                    media_files = []
                    video_url = ""
                    allowed_types = mode_cfg["file_types"]
                    if allowed_types or mode_cfg["allow_video_url"]:
                        with st.expander(L("Add evidence for this turn (optional)", "为本轮添加证据（可选）"), expanded=False):
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
                                    L("Public video URL (optional)", "公开视频 URL（可选）"),
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
                    st.success(L("Session ended. Start/reset a session whenever you want to practice another conversation.", "会话已结束。需要练习新的对话时，可以随时开始或重置会话。"))
            with workspace_col:
                render_workspace(manual_state, show_coaching=False)

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
