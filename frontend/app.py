from __future__ import annotations

import html
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

APP_VERSION = "0.8.1-tencent-multimodal"

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
        "hint": "Text-first support. Customer wording is the primary signal; screenshots may add visual evidence.",
        "affect_source": "text",
    },
    "Voice Call": {
        "icon": "🎧", "slug": "voice call",
        "hint": "Spoken support. Uploaded audio is transcribed by Hy-ASR; emotion is inferred from wording, not raw vocal tone.",
        "affect_source": "audio",
    },
    "Video + Voice": {
        "icon": "◉", "slug": "video + voice call",
        "hint": "YT-VITA analyzes uploaded video frames and its audio track; JSpace preserves that evidence separately from backend records.",
        "affect_source": "video",
    },
    "Multimodal Mix": {
        "icon": "✦", "slug": "multimodal conversation",
        "hint": "DeepSeek image/text + Hy-ASR audio transcription + YT-VITA video + backend evidence in one shared workspace.",
        "affect_source": "audio",
    },
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
.j-phone { border:1px solid rgba(108,195,255,.19); border-radius:24px; background:linear-gradient(180deg,rgba(6,12,24,.88),rgba(7,13,25,.72)); padding:.75rem .78rem 1rem; box-shadow:0 18px 55px rgba(0,0,0,.20), inset 0 0 38px rgba(51,117,216,.035); min-height:420px; max-height:650px; overflow-y:auto; }
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

/* Compact top-right utility toolbar. Each action is a true square icon button;
   dialogs replace popovers so there are no oversized chevrons or uneven controls. */
.st-key-utility_toolbar { margin-top:-.15rem; margin-bottom:.38rem; }
.st-key-utility_toolbar [data-testid="stHorizontalBlock"] { align-items:center; gap:.42rem!important; }
.st-key-utility_toolbar [data-testid="column"]:first-child { flex:1 1 auto!important; min-width:0!important; }
.st-key-utility_toolbar [data-testid="column"]:not(:first-child) {
  flex:0 0 2.45rem!important; width:2.45rem!important; min-width:2.45rem!important; max-width:2.45rem!important;
}
.st-key-utility_toolbar .stButton { margin:0!important; width:2.45rem!important; }
.st-key-utility_toolbar .stButton > button {
  width:2.40rem!important; min-width:2.40rem!important; max-width:2.40rem!important;
  height:2.40rem!important; min-height:2.40rem!important; padding:0!important;
  border-radius:12px!important; border:1px solid rgba(93,245,255,.24)!important;
  background:linear-gradient(145deg,rgba(14,29,49,.90),rgba(8,17,31,.90))!important;
  color:#DFF8FF!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 7px 20px rgba(0,0,0,.16)!important;
  display:flex!important; align-items:center!important; justify-content:center!important; transition:.16s ease!important;
}
.st-key-utility_toolbar .stButton > button:hover {
  transform:translateY(-1px); border-color:rgba(93,245,255,.55)!important;
  background:linear-gradient(145deg,rgba(18,43,65,.96),rgba(13,25,45,.96))!important;
  box-shadow:0 8px 24px rgba(58,177,220,.12),inset 0 1px 0 rgba(255,255,255,.06)!important;
}
.st-key-utility_toolbar .stButton > button:active { transform:translateY(0); }
.st-key-utility_toolbar .stButton > button p { margin:0!important; line-height:1!important; font-size:0!important; }
.st-key-utility_toolbar .stButton > button [data-testid="stIconMaterial"] {
  font-size:1.22rem!important; width:1.22rem!important; height:1.22rem!important; margin:0!important;
}
@media(max-width:700px){
  .st-key-utility_toolbar [data-testid="column"]:not(:first-child){flex-basis:2.25rem!important;width:2.25rem!important;min-width:2.25rem!important;max-width:2.25rem!important;}
  .st-key-utility_toolbar .stButton,.st-key-utility_toolbar .stButton>button{width:2.20rem!important;min-width:2.20rem!important;max-width:2.20rem!important;height:2.20rem!important;min-height:2.20rem!important;}
}
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


def _init_preferences() -> None:
    defaults = {
        "settings_speed": "Fast",
        "settings_reply": "Concise",
        "settings_scenario_ai": True,
        "settings_auto_scroll": True,
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


_init_preferences()

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


def display_domain(domain: str) -> str:
    return domain.replace("_", " ").title()


def reset_sessions() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("live_", "manual_")):
            del st.session_state[key]


# Compact utility controls in the top-right. Dialogs keep the toolbar itself
# visually minimal and avoid Streamlit popover chevrons/oversized buttons.
@st.dialog("Help", width="small")
def _help_dialog() -> None:
    st.markdown("**Quick guide**")
    st.write("Scenario Lab generates a case and reveals it turn by turn. Manual mode lets you play the customer until you end the session.")
    st.write("Multimodal Mix routes text/images to DeepSeek Vision, audio to Hy-ASR transcription, video to YT-VITA, then combines the resulting evidence with company-system data.")


@st.dialog("Share", width="small")
def _share_dialog() -> None:
    st.markdown("**Email this experience**")
    share_url = st.text_input("Public app link", value=PUBLIC_APP_URL, placeholder="https://your-app.streamlit.app", key="share_url")
    recipient = st.text_input("Recipient email", placeholder="name@example.com", key="share_email")
    if share_url and recipient and "@" in recipient:
        subject = urllib.parse.quote("Try JSpace Live")
        body = urllib.parse.quote(f"I wanted to share this JSpace customer-service experience with you:\n\n{share_url}")
        st.link_button("Email link", f"mailto:{urllib.parse.quote(recipient)}?subject={subject}&body={body}", width="stretch")
        st.caption("This opens your email app with the recipient and website link filled in; you press Send there.")
    elif recipient and "@" not in recipient:
        st.caption("Enter a valid email address.")


@st.dialog("Settings", width="small")
def _settings_dialog() -> None:
    st.markdown("**Conversation settings**")
    st.selectbox(
        "AI response profile", ["Fast", "Balanced"], key="settings_speed",
        help="Fast uses a 12-second DeepSeek attempt cap and 4 recent messages. Balanced uses a 20-second cap and 6 recent messages.",
    )
    st.selectbox("Agent reply length", ["Concise", "Standard"], key="settings_reply")
    st.toggle("Use DeepSeek to vary scenario wording", key="settings_scenario_ai", help="Turn this off for near-instant curated scenario generation.")
    st.toggle("Auto-jump to conversation after generation", key="settings_auto_scroll")
    cfg = _ai_runtime()
    st.caption(f"Text/image: {TOKENHUB_MODEL} · Audio: {TOKENHUB_AUDIO_MODEL} · Video: {TOKENHUB_VIDEO_MODEL}")
    st.caption(f"DeepSeek: {cfg['timeout_ms']/1000:.0f}s/attempt · {cfg['history_turns']} recent messages · max {cfg['max_attempts']} attempts")
    if st.button("Test DeepSeek connection", width="stretch", key="test_deepseek"):
        with st.spinner("Testing…"):
            ok, detail = probe_deepseek(api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL, timeout_s=12.0)
        if ok:
            st.success(f"Connected · {TOKENHUB_MODEL}")
        else:
            st.error(detail)
    components.html(
        """<button onclick="window.parent.print()" style="width:100%;padding:8px 12px;border-radius:10px;border:1px solid #4a7891;background:#0d1b2b;color:#eaf7ff;cursor:pointer">Print this view</button>""",
        height=46,
    )


with st.container(key="utility_toolbar"):
    spacer, u1, u2, u3, u4 = st.columns([1, .04, .04, .04, .04], gap="small", vertical_alignment="center")
    with u1:
        if st.button("\u200b", icon=":material/help:", help="Help", key="top_help", width="stretch"):
            _help_dialog()
    with u2:
        if st.button("\u200b", icon=":material/share:", help="Share", key="top_share", width="stretch"):
            _share_dialog()
    with u3:
        if st.button("\u200b", icon=":material/refresh:", help="Reset current sessions", key="top_reset", width="stretch"):
            _bump_generation_epoch()
            reset_sessions()
            st.rerun()
    with u4:
        if st.button("\u200b", icon=":material/settings:", help="Settings", key="top_settings", width="stretch"):
            _settings_dialog()

st.markdown("<div style=\"height:.65rem\"></div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="j-hero">
  <div class="j-kicker">JSPACE // MULTIMODAL SERVICE LAB</div>
  <div class="j-title">Live customer-service reasoning across text, voice, video and system evidence.</div>
  <div class="j-sub">Observe a capacity-limited shared workspace preserve task-critical signals, uncertainty and cross-modal conflicts while the support agent works toward a natural resolution.</div>
  <span class="j-pill">v{APP_VERSION}</span>
  <span class="j-pill">{'Tencent multimodal routing enabled' if AI_CONNECTED else 'Local simulation mode'}</span>
</div>
""",
    unsafe_allow_html=True,
)



def profile_html(profile: dict) -> str:
    items = [
        ("Customer", profile.get("name", "—")), ("Tenure", profile.get("tenure", "—")),
        ("Relationship", profile.get("relationship", "—")), ("Loyalty", profile.get("loyalty_tier", "—")),
        ("Contacts · 90d", profile.get("previous_contacts_90d", "—")), ("Value segment", profile.get("value_segment", "—")),
        ("Communication", profile.get("communication_style", "—")), ("Tech comfort", profile.get("tech_comfort", "—")),
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
    c1.caption(f"Patience · {profile.get('patience', 0)}/100")
    c2.progress(max(0.0, min(1.0, profile.get("trust", 0) / 100)))
    c2.caption(f"Trust in company · {profile.get('trust', 0)}/100")
    satisfaction = float(getattr(state, "customer_satisfaction", 50.0)) if state is not None else 50.0
    c3.progress(max(0.0, min(1.0, satisfaction / 100)))
    c3.caption(f"Satisfaction · {satisfaction:.0f}/100")
    c4.markdown(
        f'<div class="j-card"><div class="j-card-title">Preferred channel</div><div class="j-card-value">{html.escape(str(profile.get("preferred_channel", "—")).title())}</div></div>',
        unsafe_allow_html=True,
    )


def _emotion_html(state) -> str:
    label = (state.current_emotion or "Waiting for signal").replace("_", " ").title()
    size = "1.42rem" if len(label) <= 11 else ("1.14rem" if len(label) <= 17 else ".96rem")
    intensity = f"{state.current_emotion_intensity:.0%}" if state.current_emotion else "—"
    return f'''<div class="j-emotion"><div class="j-emotion-label">Customer affect</div><div class="j-emotion-value" style="font-size:{size}">{html.escape(label)}</div><div class="j-card-meta">Affect intensity · {intensity}</div></div>'''


def concept_rows(state) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Concept": c.name.replace("_", " ").title(), "Value": c.value, "Status": c.status,
            "Sources": ", ".join(c.sources), "Priority": round(c.score, 2), "Confidence": round(c.confidence, 2),
        }
        for c in state.active_concepts
    ])


def render_workspace(state, *, show_coaching: bool = True) -> None:
    st.markdown("#### Live JSpace")
    if show_coaching:
        top1, top2 = st.columns([1.35, .65])
        with top1:
            st.markdown(
                f'''<div class="j-card j-next"><div class="j-card-title">Recommended next move</div><div class="j-card-value">{html.escape(state.recommended_action or "Gather the first customer signal")}</div><div class="j-card-meta">Support coaching cue · the next action most likely to advance resolution.</div></div>''',
                unsafe_allow_html=True,
            )
        with top2:
            st.markdown(_emotion_html(state), unsafe_allow_html=True)
    else:
        st.markdown(_emotion_html(state), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.caption(f"Active concepts · {len(state.active_concepts)} / {state.config.capacity_k}")
    c2.caption(f"Signal conflicts · {len(state.conflicts)}")
    c3.caption(f"Session phase · {state.session_phase.title()}")

    if state.conflicts:
        for conflict in state.conflicts:
            st.markdown(
                f'<div class="j-card j-conflict"><div class="j-card-title">{html.escape(conflict.severity.upper())} SIGNAL CONFLICT</div><div class="j-card-value">{html.escape(conflict.description)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("##### Active workspace concepts")
    if not state.active_concepts:
        st.info("JSpace will populate as customer, media, and company evidence arrive.")
    else:
        for c in state.active_concepts:
            sources = " · ".join(s.title() for s in c.sources)
            st.markdown(
                f'''<div class="j-card j-concept {html.escape(c.status)}"><div class="j-card-title">{html.escape(c.name.replace('_',' ').title())}</div><div class="j-card-value">{html.escape(str(c.value))}</div><div class="j-card-meta">{html.escape(sources)} · priority {c.score:.2f} · {html.escape(c.status)}</div></div>''',
                unsafe_allow_html=True,
            )

    with st.expander("Evidence & provenance", expanded=False):
        df = concept_rows(state)
        if df.empty:
            st.caption("No active evidence yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Priority is a workspace ranking signal, not an accuracy or satisfaction score.")


def _message_html(row: dict, channel_label: str) -> str:
    role = row.get("role", "customer")
    text = row.get("text", "")
    if role == "agent":
        who = "JSpace Support Agent"
        provider = str(row.get("provider") or "").strip()
        if provider:
            if provider.lower().startswith("deepseek"):
                meta = "AI provider · " + provider
            elif "fallback" in provider.lower() or "simulation" in provider.lower():
                meta = "Backup responder · " + provider
            else:
                meta = provider
        else:
            meta = ""
    else:
        who = "Customer"
        details = []
        if row.get("emotion"):
            details.append(str(row["emotion"]).replace("_", " ").title())
        if isinstance(row.get("emotion_intensity"), (float, int)):
            details.append(f"{row['emotion_intensity']:.0%} affect")
        if row.get("nonverbal_cue") and channel_label != "Text Messages":
            details.append(str(row["nonverbal_cue"]))
        attachments = row.get("attachments", [])
        if attachments:
            details.append("media: " + ", ".join(x.get("name", "attachment") for x in attachments))
        meta = " · ".join(details)
    meta_html = f'<div class="j-msg-meta">{html.escape(str(meta))}</div>' if meta else ""
    return f'''<div class="j-msg-row {role}"><div class="j-msg {role}"><div style="font-size:.63rem;color:#81A5C5;font-weight:700;margin-bottom:.24rem">{html.escape(who)}</div>{html.escape(str(text))}{meta_html}</div></div>'''


def _conversation_html(transcript: list[dict], channel_label: str, *, typing: bool = False) -> str:
    info = CHANNELS[channel_label]
    messages = [_message_html(row, channel_label) for row in transcript]
    if typing:
        messages.append('<div class="j-msg-row agent"><div class="j-msg agent"><div class="j-typing">Support Agent is typing…</div></div></div>')
    if not messages:
        messages.append('<div class="j-card-meta" style="padding:.9rem">Conversation has not started yet.</div>')
    return f'''<div class="j-phone"><div class="j-phone-head"><div><div class="j-channel-name">{info['icon']} {html.escape(channel_label)}</div><div class="j-channel-meta">{html.escape(info['hint'])}</div></div><div class="j-channel-meta"><span class="j-live-dot"></span>LIVE SESSION</div></div>{''.join(messages)}</div>'''


def render_conversation(transcript: list[dict], channel_label: str, *, typing: bool = False, slot=None) -> None:
    target = slot or st.empty()
    target.markdown(_conversation_html(transcript, channel_label, typing=typing), unsafe_allow_html=True)


def make_responder(channel_label: str, media: list[dict] | None = None):
    def responder(state, profile, domain):
        cfg = _ai_runtime()
        return generate_support_reply(
            state, profile, domain, api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL,
            fallback=state.last_response, channel=CHANNELS[channel_label]["slug"], media=media,
            timeout_s=cfg["timeout_ms"] / 1000.0, max_attempts=cfg["max_attempts"],
            history_turns=cfg["history_turns"], max_output_tokens=cfg["max_output_tokens"],
            reply_sentences=cfg["reply_sentences"],
        )
    return responder


def stream_agent_reply(state, profile, domain: str, channel_label: str, conversation_slot, *, media=None, epoch: int | None = None):
    """Stream the support reply into the existing phone UI and return the final text/provider."""
    cfg = _ai_runtime()
    final_text, final_provider = state.last_response or "I can help with that.", "Local simulation"
    got_text = False
    for partial, provider, done in stream_support_reply(
        state, profile, domain, api_key=TOKENHUB_API_KEY, model=TOKENHUB_MODEL, base_url=TOKENHUB_BASE_URL,
        fallback=state.last_response, channel=CHANNELS[channel_label]["slug"], media=media,
        timeout_s=cfg["timeout_ms"] / 1000.0, max_attempts=cfg["max_attempts"],
        history_turns=cfg["history_turns"], max_output_tokens=cfg["max_output_tokens"],
        reply_sentences=cfg["reply_sentences"],
    ):
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
        return CUSTOMER_STARTERS.get(domain, "I need help with an issue on my account. Can you check the current system status?")
    if state.session_ended:
        return ""
    if state.conflicts:
        return "What I'm seeing still doesn't match what you're telling me. Can you verify which system is authoritative and explain the mismatch?"
    if state.session_phase == "resolved":
        return "Thanks. Can you confirm there isn't anything else I need to do on my side?"
    if state.recommended_action_code == "act_on_root_cause":
        return "Can you explain what you found and what you can do now to actually fix it?"
    return "Can you tell me what you've verified so far and what the next concrete step is?"


def _use_manual_suggestion(suggestion: str) -> None:
    """Prefill the manual chat widget on the next rerun."""
    st.session_state["manual_chat_input"] = suggestion


def render_start_here(domains: list[str]) -> None:
    st.markdown("## How to use JSpace Live")
    st.markdown("Explore a generated service interaction or become the customer yourself. Customer messages appear immediately while the support agent is typing. The active workspace is deliberately small so you can see what evidence survives, what conflicts, and what changes the next response.")
    a, b = st.columns(2)
    with a:
        st.markdown('''<div class="j-card j-case"><div class="j-card-title">MODE 01</div><div class="j-card-value">Scenario Lab</div><div class="j-card-meta">Generate a controlled case. The customer appears one turn at a time, the support agent responds live, and the conversation only closes after confirmed resolution and a normal final check for other concerns.</div></div>''', unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="j-card j-case"><div class="j-card-title">MODE 02</div><div class="j-card-value">Manual Multimodal AI</div><div class="j-card-meta">You play the customer for as many turns as needed. Attach screenshots, voice clips or video in multimodal channels, or end the session whenever you are finished.</div></div>''', unsafe_allow_html=True)

    st.markdown("### What the JSpace pipeline is doing")
    nodes = [
        ("01", "Signals", "Customer text, DeepSeek image evidence, Hy-ASR audio transcripts, YT-VITA video evidence, and company-system events."),
        ("02", "Concepts", "Signals become compact, traceable task-relevant concepts."),
        ("03", "Conflict engine", "Contradictions remain explicit instead of being averaged away."),
        ("04", "Top-K JSpace", "Only a few concepts stay active; v0.8 uses K=3–6 to keep the workspace genuinely compact."),
        ("05", "Support response", "The agent reasons over the active state and recent conversation to move toward resolution."),
    ]
    node_html = "".join(f'<div class="j-node"><div class="j-node-num">NODE {n}</div><div class="j-node-name">{name}</div><div class="j-node-desc">{desc}</div></div>' for n, name, desc in nodes)
    st.markdown(f'<div class="j-node-grid">{node_html}</div>', unsafe_allow_html=True)

    st.markdown("### Key controls and signals")
    terms = [
        ("JSpace capacity (K)", "Maximum number of concepts allowed in the active workspace. Smaller K makes selection stricter; it does not directly control model latency."),
        ("Customer affect intensity", "Scenario Lab can simulate nonverbal affect. In Manual mode, affect is inferred from typed/transcribed wording; Hy-ASR is transcription-only and does not classify vocal emotion."),
        ("Satisfaction", "A dynamic interaction-quality signal that rises with useful progress/resolution and falls when the exchange remains confusing or unhelpful."),
        ("Priority", "The ranking used to decide what survives Top-K: relevance, confidence, conflict importance and recency."),
        ("Evidence & provenance", "Where each active concept came from — text, audio, image/video, backend systems, or derived reasoning."),
        ("Researcher view", "Hidden simulated company truth and provider diagnostics. Closed by default because a real customer would not see it."),
    ]
    cols = st.columns(3)
    for i, (name, desc) in enumerate(terms):
        with cols[i % 3]:
            st.markdown(f'<div class="j-domain"><strong>{html.escape(name)}</strong><span>{html.escape(desc)}</span></div>', unsafe_allow_html=True)

    st.markdown("### Customer-service domains")
    cols = st.columns(3)
    for i, domain in enumerate(domains):
        with cols[i % 3]:
            st.markdown(f'<div class="j-domain"><strong>{html.escape(display_domain(domain))}</strong><span>{html.escape(DOMAIN_DESCRIPTIONS.get(domain, "Customer-service case."))}</span></div>', unsafe_allow_html=True)


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
    ["◎ Start Here", "✦ Scenario Lab", "◈ Manual Multimodal AI"],
    on_change=_on_main_tab_change,
    key="main_tabs",
)

if start_tab.open:
    with start_tab:
        render_start_here(domains)

if scenario_tab.open:
    with scenario_tab:
        st.markdown("## Scenario Lab")
        control1, control2, control3 = st.columns([1.25, 1, 1])
        with control1:
            domain_label = st.selectbox("Domain", ["Random"] + [display_domain(d) for d in domains], key="scenario_domain")
            scenario_domain = "random" if domain_label == "Random" else domain_label.lower().replace(" ", "_")
            if scenario_domain != "random":
                st.caption(DOMAIN_DESCRIPTIONS.get(scenario_domain, ""))
        with control2:
            channel_label = st.selectbox("Conversation channel", list(CHANNELS), index=0, key="scenario_channel")
            st.caption(CHANNELS[channel_label]["hint"])
        with control3:
            scenario_k = st.slider("JSpace capacity K", 3, 6, 4, key="scenario_k")
            seed_text = st.text_input("Optional seed", placeholder="blank = new case", key="scenario_seed")
            seed = int(seed_text) if seed_text.strip().isdigit() else None

        if st.button("Generate scenario", type="primary", use_container_width=True, key="generate_scenario"):
            _bump_generation_epoch()
            epoch = int(st.session_state.generation_epoch)
            with st.spinner("Building a customer case…"):
                scenario = generate_scenario(ScenarioControls(domain=scenario_domain, seed=seed))
                scenario_provider = "Curated scenario · instant"
                if AI_CONNECTED and st.session_state.get("settings_scenario_ai", True):
                    cfg = _ai_runtime()
                    scenario, scenario_provider = enhance_scenario_with_deepseek(
                        scenario,
                        api_key=TOKENHUB_API_KEY,
                        model=TOKENHUB_MODEL,
                        base_url=TOKENHUB_BASE_URL,
                        channel=CHANNELS[channel_label]["slug"],
                        timeout_s=cfg["scenario_timeout_ms"] / 1000.0,
                    )
                scenario = prepare_scenario_for_channel(scenario, channel_label)
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
            st.info("Generate a scenario first. The case brief will appear here, then you can start the live conversation.")
        else:
            st.markdown('<div id="scenario-live-anchor"></div>', unsafe_allow_html=True)
            if st.session_state.pop("scenario_scroll_pending", False) and st.session_state.get("settings_auto_scroll", True):
                _scroll_to("scenario-live-anchor")

            st.markdown(
                f'''<div class="j-card j-case"><div class="j-card-title">CASE BRIEF · {html.escape(display_domain(scenario.domain))}</div><div class="j-card-value">{html.escape(scenario.title)}</div><div class="j-card-meta">{html.escape(scenario.problem_summary or scenario.steps[0].customer_turn.text)} · {html.escape(CHANNELS[live_channel]['hint'])}</div></div>''',
                unsafe_allow_html=True,
            )
            render_profile(scenario.customer_profile, state)

            chat_col, workspace_col = st.columns([1.14, .86], gap="large")
            with chat_col:
                conversation_slot = st.empty()
                render_conversation(state.transcript, live_channel, slot=conversation_slot)

                if not st.session_state.get("live_started", False) and not state.session_ended:
                    start_c, end_c = st.columns([3, 1])
                    if start_c.button("Start conversation ▶", type="primary", use_container_width=True, key="start_live"):
                        st.session_state.live_started = True
                        ok = process_scenario_turn(scenario, state, 0, live_channel, conversation_slot)
                        if ok:
                            st.session_state.live_next_step = 1
                            st.session_state.live_state = state
                        st.rerun()
                    if end_c.button("End session", use_container_width=True, key="end_scenario_before_start"):
                        _bump_generation_epoch()
                        end_scenario_session(state)
                        st.session_state.live_user_ended = True
                        st.session_state.live_state = state
                        st.rerun()
                elif not state.session_ended and next_step < len(scenario.steps):
                    next_c, end_c = st.columns([3, 1])
                    if next_c.button("Continue conversation →", type="primary", use_container_width=True, key=f"continue_scenario_{next_step}"):
                        ok = process_scenario_turn(scenario, state, next_step, live_channel, conversation_slot)
                        if ok:
                            st.session_state.live_next_step = next_step + 1
                            st.session_state.live_state = state
                        st.rerun()
                    if end_c.button("End session", use_container_width=True, key=f"end_scenario_{next_step}"):
                        _bump_generation_epoch()
                        end_scenario_session(state)
                        st.session_state.live_user_ended = True
                        st.session_state.live_state = state
                        st.rerun()
                elif state.session_ended:
                    if st.session_state.get("live_user_ended", False):
                        st.info("Practice session ended by the user. The app does not claim that the unresolved case was resolved.")
                    else:
                        st.success("Conversation closed naturally after confirmed resolution and the customer's final check-in.")

            with workspace_col:
                render_workspace(state, show_coaching=True)

            with st.expander("Researcher view · scenario ground truth", expanded=False):
                st.write("**Domain:**", display_domain(scenario.domain))
                st.write("**Problem summary:**", scenario.problem_summary)
                st.write("**Random conflict present:**", scenario.expected_conflict)
                st.write("**Hidden ground truth:**", scenario.hidden_ground_truth)
                st.write("**Scenario language source:**", st.session_state.get("live_scenario_provider", "Curated scenario"))
                st.write("**Seed:**", scenario.seed)
                st.write("**Planned customer turns:**", len(scenario.steps))
                providers = [r.get("provider") for r in state.transcript if r.get("role") == "agent"]
                st.write("**Agent providers:**", providers)

if manual_tab.open:
    with manual_tab:
        st.markdown("## Manual Multimodal AI")
        m1, m2, m3 = st.columns([1.25, 1, 1])
        with m1:
            manual_domain_label = st.selectbox("Domain", [display_domain(d) for d in domains], key="manual_domain")
            manual_domain = manual_domain_label.lower().replace(" ", "_")
            st.caption(DOMAIN_DESCRIPTIONS.get(manual_domain, ""))
        with m2:
            manual_channel = st.selectbox("Channel", list(CHANNELS), index=3, key="manual_channel")
            st.caption(CHANNELS[manual_channel]["hint"])
        with m3:
            manual_k = st.slider("JSpace capacity K", 3, 6, 4, key="manual_k")
            start_manual = st.button("Start / reset session", type="primary", use_container_width=True)

        if start_manual:
            _bump_generation_epoch()
            profile, backend_events = generate_manual_context(manual_domain)
            st.session_state.manual_state_v06 = new_manual_state(capacity_k=manual_k, backend_events=backend_events, profile=profile)
            st.session_state.manual_profile_v06 = profile
            st.session_state.manual_domain_v06 = manual_domain
            st.session_state.manual_channel_v06 = manual_channel
            st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
            st.rerun()

        manual_state = st.session_state.get("manual_state_v06")
        manual_profile = st.session_state.get("manual_profile_v06")
        active_manual_domain = st.session_state.get("manual_domain_v06", manual_domain)
        active_manual_channel = st.session_state.get("manual_channel_v06", manual_channel)

        if manual_state and manual_profile:
            st.markdown(
                f'''<div class="j-card j-case"><div class="j-card-title">PRACTICE CASE · {html.escape(display_domain(active_manual_domain))}</div><div class="j-card-value">You are the customer. Continue for as many turns as needed.</div><div class="j-card-meta">The company record is simulated automatically. In multimodal channels you can attach media that may support or contradict what you say.</div></div>''',
                unsafe_allow_html=True,
            )
            render_profile(manual_profile, manual_state)

            chat_col, workspace_col = st.columns([1.14, .86], gap="large")
            with chat_col:
                conversation_slot = st.empty()
                render_conversation(manual_state.transcript, active_manual_channel, slot=conversation_slot)

                if not manual_state.session_ended:
                    suggestion = suggested_customer_prompt(active_manual_domain, manual_state)

                    media_types = {
                        "Text Messages": ["png", "jpg", "jpeg", "webp"],
                        "Voice Call": ["mp3", "wav", "m4a", "ogg"],
                        "Video + Voice": ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "mp4", "mov", "avi", "webm"],
                        "Multimodal Mix": ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "mp4", "mov", "avi", "webm"],
                    }
                    media_files = st.file_uploader(
                        "Attach evidence for this turn (optional)",
                        type=media_types[active_manual_channel], accept_multiple_files=True,
                        key=f"manual_media_{st.session_state.get('manual_media_key', 0)}",
                        help="Images are analyzed by DeepSeek Vision, audio is transcribed by Hy-ASR, and video is analyzed by YT-VITA. All outputs become explicit JSpace evidence.",
                    )
                    if media_files:
                        st.caption("Attached: " + ", ".join(f.name for f in media_files))

                    video_url = ""
                    if active_manual_channel in {"Video + Voice", "Multimodal Mix"}:
                        video_url = st.text_input(
                            "Public video URL (optional)",
                            key=f"manual_video_url_{st.session_state.get('manual_media_key', 0)}",
                            placeholder="https://.../clip.mp4",
                            help="YT-VITA officially documents video_url input. A public MP4/MOV/AVI/WebM URL is the most reliable path; local video uploads use a best-effort data URL.",
                        )

                    # Keep the chat composer directly below the conversation. A single-line
                    # text input intentionally submits on Enter, which feels closer to live chat.
                    input_label = "Message the support agent as the customer"
                    with st.form("manual_turn_form", clear_on_submit=True):
                        prompt = st.text_input(
                            input_label,
                            key="manual_chat_input",
                            placeholder="Type a customer message and press Enter…",
                            help="Press Enter or click Send message. Your message appears immediately before the support agent starts generating its reply.",
                        )
                        send = st.form_submit_button("Send message", type="primary", use_container_width=True)

                    suggest_left, suggest_right = st.columns([4.2, 1.3], gap="small")
                    with suggest_left:
                        st.markdown(
                            f'<div class="j-suggest">Suggested customer prompt: {html.escape(suggestion)}</div>',
                            unsafe_allow_html=True,
                        )
                    with suggest_right:
                        st.button(
                            "Use suggested prompt",
                            key="use_manual_suggestion",
                            use_container_width=True,
                            on_click=_use_manual_suggestion,
                            args=(suggestion,),
                        )

                    end_now = st.button("End session", use_container_width=True, key="manual_end_session")

                    if send and (prompt.strip() or media_files or video_url.strip()):
                        customer_text = prompt.strip() or "[Customer attached media evidence for this turn.]"
                        ok = process_manual_turn(
                            manual_state, manual_profile, active_manual_domain, active_manual_channel,
                            customer_text, media_files, conversation_slot, video_url=video_url,
                        )
                        if ok:
                            st.session_state.manual_state_v06 = manual_state
                            st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
                        st.rerun()
                    if end_now:
                        _bump_generation_epoch()
                        end_manual_session(manual_state)
                        st.session_state.manual_state_v06 = manual_state
                        st.rerun()
                else:
                    st.success("Session ended. Start/reset a session whenever you want to practice another conversation.")
            with workspace_col:
                render_workspace(manual_state, show_coaching=False)

            with st.expander("Researcher view · simulated company context", expanded=False):
                st.write("**Domain:**", display_domain(active_manual_domain))
                st.write("**Channel:**", active_manual_channel)
                st.write("**Company-system events:**", manual_state.backend_history)
                st.write("**AI connected:**", AI_CONNECTED)
                st.write("**Provider:**", "Tencent TokenHub")
                st.write("**Text + image model:**", TOKENHUB_MODEL)
                st.write("**Audio transcription:**", TOKENHUB_AUDIO_MODEL)
                st.write("**Video understanding:**", TOKENHUB_VIDEO_MODEL)
                st.write("**Agent providers:**", [r.get("provider") for r in manual_state.transcript if r.get("role") == "agent"])
        else:
            st.info("Choose a domain/channel and start a session. Multimodal Mix gives the richest demonstration of conflicting evidence across modalities.")

st.markdown("---")
st.caption("JSpace Live · capacity-limited, conflict-aware multimodal customer-service research experience")
