from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ai_provider import (  # noqa: E402
    DEFAULT_MODEL,
    analyze_media_for_jspace,
    enhance_scenario_with_gemini,
    generate_support_reply,
)
from backend.app.engine import merge_concepts, refresh_state  # noqa: E402
from backend.app.scenario_generator import generate_manual_context, generate_scenario, list_domains  # noqa: E402
from backend.app.schemas import ScenarioControls  # noqa: E402
from backend.app.simulator import (  # noqa: E402
    apply_scenario_step,
    manual_customer_turn,
    new_manual_state,
    new_scenario_state,
)

APP_VERSION = "0.5.0-gemini-multimodal"

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
    "Text Messages": {"icon": "💬", "slug": "text messages", "hint": "Phone-style asynchronous chat. Screenshots can be attached."},
    "Voice Call": {"icon": "🎧", "slug": "voice call", "hint": "Typed text represents spoken words; JSpace also tracks inferred vocal affect."},
    "Video + Voice": {"icon": "◉", "slug": "video + voice call", "hint": "Call transcript plus optional image/video evidence and vocal cues."},
    "Multimodal Mix": {"icon": "✦", "slug": "multimodal conversation", "hint": "Combine typed/spoken content with image, audio, and video evidence in one turn."},
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
  --j-text:#EDF7FF; --j-muted:#8EA6C0; --j-green:#7DFFBD; --j-orange:#FFC178;
}
.stApp {
  background:
    radial-gradient(circle at 8% 4%, rgba(75,126,255,.20), transparent 30%),
    radial-gradient(circle at 94% 14%, rgba(177,116,255,.15), transparent 27%),
    radial-gradient(circle at 55% 88%, rgba(35,224,226,.07), transparent 31%),
    linear-gradient(180deg,#060A14 0%,#040710 58%,#060A12 100%);
  color:var(--j-text);
}
.block-container { max-width:1500px; padding-top:1.25rem; padding-bottom:4rem; }
[data-testid="stHeader"] { background:rgba(0,0,0,0); }
.j-hero { padding:1.45rem 1.65rem; border:1px solid var(--j-border); border-radius:22px; background:linear-gradient(135deg,rgba(16,31,59,.90),rgba(8,15,31,.76)); box-shadow:0 20px 70px rgba(0,0,0,.31), inset 0 1px 0 rgba(255,255,255,.04); position:relative; overflow:hidden; margin-bottom:1rem; }
.j-hero:before { content:""; position:absolute; top:0; left:-25%; width:55%; height:2px; background:linear-gradient(90deg,transparent,var(--j-cyan),var(--j-violet),transparent); animation:scan 5s linear infinite; }
@keyframes scan { from{transform:translateX(0)} to{transform:translateX(230%)} }
.j-kicker { color:var(--j-cyan); letter-spacing:.20em; font-size:.70rem; font-weight:800; }
.j-title { font-size:2.28rem; line-height:1.08; font-weight:760; margin:.34rem 0 .42rem; color:#F8FBFF; }
.j-sub { color:var(--j-muted); max-width:980px; font-size:.98rem; line-height:1.55; }
.j-pill { display:inline-block; padding:.20rem .53rem; border-radius:999px; border:1px solid rgba(97,244,255,.25); background:rgba(97,244,255,.07); color:#C5FBFF; font-size:.72rem; margin:.6rem .35rem 0 0; }
.j-ai-live { border-color:rgba(125,255,189,.30); color:#AFFFF0; background:rgba(63,175,120,.10); }
.j-ai-local { border-color:rgba(255,193,120,.28); color:#FFD8A9; background:rgba(154,98,38,.10); }
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
.j-emotion { border:1px solid rgba(177,116,255,.25); background:rgba(63,38,90,.18); border-radius:14px; padding:.68rem .82rem; min-height:82px; }
.j-emotion-label { color:#9E86C0; font-size:.64rem; text-transform:uppercase; letter-spacing:.09em; }
.j-emotion-value { color:#F4E9FF; font-weight:750; line-height:1.06; margin-top:.24rem; overflow-wrap:anywhere; }
.j-phone { border:1px solid rgba(108,195,255,.19); border-radius:24px; background:linear-gradient(180deg,rgba(6,12,24,.88),rgba(7,13,25,.72)); padding:.75rem .78rem 1rem; box-shadow:0 18px 55px rgba(0,0,0,.20), inset 0 0 38px rgba(51,117,216,.035); min-height:460px; }
.j-phone-head { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(109,173,220,.12); padding:.38rem .32rem .70rem; margin-bottom:.62rem; }
.j-channel-name { color:#EAF7FF; font-size:.90rem; font-weight:700; } .j-channel-meta { color:#6F8AA8; font-size:.69rem; }
.j-live-dot { display:inline-block; width:7px; height:7px; background:#6DFFB3; border-radius:50%; box-shadow:0 0 10px rgba(109,255,179,.8); margin-right:.35rem; animation:pulse 1.8s infinite; }
@keyframes pulse { 50%{opacity:.45; transform:scale(.85)} }
.j-msg-row { display:flex; margin:.52rem .18rem; }
.j-msg-row.customer { justify-content:flex-end; } .j-msg-row.agent { justify-content:flex-start; }
.j-msg { max-width:78%; border-radius:17px; padding:.66rem .82rem; line-height:1.45; font-size:.91rem; border:1px solid rgba(128,178,223,.12); box-shadow:0 8px 20px rgba(0,0,0,.10); }
.j-msg.customer { background:linear-gradient(135deg,rgba(43,95,167,.78),rgba(56,71,149,.74)); color:#F5FAFF; border-bottom-right-radius:5px; }
.j-msg.agent { background:rgba(17,28,47,.88); color:#EDF6FF; border-bottom-left-radius:5px; }
.j-msg-meta { color:#8BA6C2; font-size:.65rem; margin-top:.35rem; line-height:1.35; }
.j-typing { color:#6CEAFF; letter-spacing:.18em; font-size:.85rem; }
.j-node-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.55rem; margin:.6rem 0 1rem; }
.j-node { border:1px solid rgba(95,207,255,.17); background:linear-gradient(150deg,rgba(17,33,57,.68),rgba(10,18,34,.62)); border-radius:15px; padding:.8rem; min-height:118px; position:relative; }
.j-node:after { content:""; position:absolute; right:-.44rem; top:48%; width:.32rem; height:1px; background:#4CCDEA; opacity:.65; }
.j-node:last-child:after { display:none; }
.j-node-num { color:var(--j-cyan); font-size:.65rem; letter-spacing:.12em; } .j-node-name { font-weight:720; color:#EEF7FF; margin:.25rem 0; } .j-node-desc { color:#8EA6C0; font-size:.76rem; line-height:1.4; }
.j-domain { border:1px solid rgba(126,168,214,.13); border-radius:12px; padding:.70rem .78rem; background:rgba(10,19,34,.49); min-height:98px; margin:.3rem 0; }
.j-domain strong { color:#DDF3FF; } .j-domain span { color:#829AB5; font-size:.76rem; line-height:1.35; display:block; margin-top:.22rem; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,var(--j-blue),var(--j-violet),var(--j-pink)); }
.stTabs [data-baseweb="tab-list"] { gap:.42rem; } .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.46rem .88rem; background:rgba(11,21,39,.62); }
.stButton > button { border-radius:12px; border:1px solid rgba(93,245,255,.23); }
[data-testid="stFileUploaderDropzone"] { background:rgba(8,17,32,.58); border-color:rgba(93,245,255,.18); }
hr { border-color:rgba(140,175,215,.12)!important; }
@media(max-width:1000px){ .j-profile-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.j-node-grid{grid-template-columns:1fr 1fr}.j-node:after{display:none}.j-title{font-size:1.72rem}.j-msg{max-width:92%} }
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


GEMINI_API_KEY = _secret("GEMINI_API_KEY")
GEMINI_MODEL = _secret("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
AI_CONNECTED = bool(GEMINI_API_KEY)

st.markdown(
    f"""
<div class="j-hero">
  <div class="j-kicker">JSPACE // MULTIMODAL SERVICE LAB</div>
  <div class="j-title">A live, capacity-limited workspace for customer-service reasoning.</div>
  <div class="j-sub">Explore how text, voice affect, visual evidence, media and authoritative company signals compete for limited workspace capacity — while a Gemini-powered agent tries to resolve the customer's problem naturally.</div>
  <span class="j-pill">v{APP_VERSION}</span>
  <span class="j-pill {'j-ai-live' if AI_CONNECTED else 'j-ai-local'}">{'Gemini connected · ' + html.escape(GEMINI_MODEL) if AI_CONNECTED else 'Local fallback · add GEMINI_API_KEY for live Gemini'}</span>
</div>
""",
    unsafe_allow_html=True,
)


def display_domain(domain: str) -> str:
    return domain.replace("_", " ").title()


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


def render_profile(profile: dict) -> None:
    st.markdown(profile_html(profile), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.progress(max(0.0, min(1.0, profile.get("patience", 0) / 100)))
    c1.caption(f"Patience · {profile.get('patience', 0)}/100")
    c2.progress(max(0.0, min(1.0, profile.get("trust", 0) / 100)))
    c2.caption(f"Trust in company · {profile.get('trust', 0)}/100")
    c3.markdown(
        f'<div class="j-card"><div class="j-card-title">Preferred channel</div><div class="j-card-value">{html.escape(str(profile.get("preferred_channel", "—")).title())}</div></div>',
        unsafe_allow_html=True,
    )


def _emotion_html(state) -> str:
    label = (state.current_emotion or "Waiting for signal").replace("_", " ").title()
    size = "1.48rem" if len(label) <= 12 else ("1.20rem" if len(label) <= 20 else "1.02rem")
    intensity = f"{state.current_emotion_intensity:.0%}" if state.current_emotion else "—"
    return f'''<div class="j-emotion"><div class="j-emotion-label">Customer affect</div><div class="j-emotion-value" style="font-size:{size}">{html.escape(label)}</div><div class="j-card-meta">Affect intensity · {intensity}</div></div>'''


def concept_rows(state) -> pd.DataFrame:
    rows = []
    for c in state.active_concepts:
        rows.append({
            "Concept": c.name.replace("_", " ").title(), "Value": c.value, "Status": c.status,
            "Sources": ", ".join(c.sources), "Priority": round(c.score, 2), "Confidence": round(c.confidence, 2),
        })
    return pd.DataFrame(rows)


def render_workspace(state) -> None:
    st.markdown("#### Live JSpace")
    top1, top2 = st.columns([1.35, .65])
    with top1:
        st.markdown(
            f'''<div class="j-card j-next"><div class="j-card-title">Recommended next move</div><div class="j-card-value">{html.escape(state.recommended_action or "Gather the first customer signal")}</div><div class="j-card-meta">Agent coaching cue · this is the next action most likely to advance resolution.</div></div>''',
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown(_emotion_html(state), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.caption(f"Workspace capacity · {len(state.active_concepts)} / {state.config.capacity_k} active concepts")
    c2.caption(f"Detected conflicts · {len(state.conflicts)}")
    if state.current_emotion:
        st.progress(max(0.0, min(1.0, state.current_emotion_intensity)))

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

    with st.expander("Evidence & provenance", expanded=True):
        df = concept_rows(state)
        if df.empty:
            st.caption("No active evidence yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Priority combines task relevance, confidence, conflict importance and recency. It is not an accuracy score.")


def _message_html(row: dict, channel_label: str, partial_text: str | None = None) -> str:
    role = row.get("role", "customer")
    text = partial_text if partial_text is not None else row.get("text", "")
    if role == "agent":
        who = "JSpace Support Agent"
        meta = row.get("provider", "")
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
    return f'''<div class="j-msg-row {role}"><div class="j-msg {role}"><div style="font-size:.63rem;color:#81A5C5;font-weight:700;margin-bottom:.24rem">{html.escape(who)}</div>{html.escape(str(text))}<div class="j-msg-meta">{html.escape(str(meta))}</div></div></div>'''


def _conversation_html(transcript: list[dict], channel_label: str, *, limit: int | None = None,
                       partial_index: int | None = None, partial_text: str | None = None,
                       typing: bool = False) -> str:
    info = CHANNELS[channel_label]
    rows = transcript if limit is None else transcript[:limit]
    messages = []
    for i, row in enumerate(rows):
        part = partial_text if partial_index == i else None
        messages.append(_message_html(row, channel_label, part))
    if typing:
        messages.append('<div class="j-msg-row agent"><div class="j-msg agent"><span class="j-typing">● ● ●</span><div class="j-msg-meta">Agent is composing a response</div></div></div>')
    if not messages:
        messages.append('<div class="j-card-meta" style="padding:.9rem">Conversation has not started yet.</div>')
    return f'''<div class="j-phone"><div class="j-phone-head"><div><div class="j-channel-name">{info['icon']} {html.escape(channel_label)}</div><div class="j-channel-meta">{html.escape(info['hint'])}</div></div><div class="j-channel-meta"><span class="j-live-dot"></span>LIVE SESSION</div></div>{''.join(messages)}</div>'''


def render_conversation(transcript: list[dict], channel_label: str, animate_from: int | None = None) -> None:
    placeholder = st.empty()
    if animate_from is None or animate_from >= len(transcript):
        placeholder.markdown(_conversation_html(transcript, channel_label), unsafe_allow_html=True)
        return

    for i in range(animate_from, len(transcript)):
        row = transcript[i]
        text = str(row.get("text", ""))
        words = text.split()
        if row.get("role") == "agent":
            placeholder.markdown(_conversation_html(transcript, channel_label, limit=i, typing=True), unsafe_allow_html=True)
            time.sleep(0.22)
        if not words:
            placeholder.markdown(_conversation_html(transcript, channel_label, limit=i + 1), unsafe_allow_html=True)
            continue
        steps = min(12, max(4, len(words)))
        chunk = max(1, (len(words) + steps - 1) // steps)
        for end in range(chunk, len(words) + chunk, chunk):
            partial = " ".join(words[: min(end, len(words))])
            placeholder.markdown(
                _conversation_html(transcript, channel_label, limit=i + 1, partial_index=i, partial_text=partial),
                unsafe_allow_html=True,
            )
            time.sleep(0.03 if row.get("role") == "customer" else 0.04)
    placeholder.markdown(_conversation_html(transcript, channel_label), unsafe_allow_html=True)


def make_responder(channel_label: str, media: list[dict] | None = None):
    def responder(state, profile, domain):
        return generate_support_reply(
            state, profile, domain, api_key=GEMINI_API_KEY, model=GEMINI_MODEL,
            fallback=state.last_response, channel=CHANNELS[channel_label]["slug"], media=media,
        )
    return responder


def read_uploaded_media(files) -> tuple[list[dict], list[dict]]:
    full, display = [], []
    for f in files or []:
        data = f.getvalue()
        mime = f.type or "application/octet-stream"
        full.append({"name": f.name, "mime_type": mime, "data": data})
        display.append({"name": f.name, "mime_type": mime})
    return full, display


def render_start_here(domains: list[str]) -> None:
    st.markdown("## How to use JSpace Live")
    st.markdown("This MVP has two ways to explore the same research idea: let the system generate a controlled customer-service case, or become the customer yourself and converse with the AI agent.")
    a, b = st.columns(2)
    with a:
        st.markdown('''<div class="j-card j-case"><div class="j-card-title">MODE 01</div><div class="j-card-value">Live Scenario Simulator</div><div class="j-card-meta">Choose a domain and communication channel. A customer profile and problem are generated, then the case unfolds one turn at a time. If Gemini is connected, it also rewrites the case to make the wording more varied and realistic.</div></div>''', unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="j-card j-case"><div class="j-card-title">MODE 02</div><div class="j-card-value">Manual Multimodal AI</div><div class="j-card-meta">You play the customer. Type messages and optionally attach image, audio or video evidence. Gemini 3.7 Flash receives the active JSpace plus your media and replies as the support agent.</div></div>''', unsafe_allow_html=True)

    st.markdown("### What the JSpace pipeline is doing")
    nodes = [
        ("01", "Signals", "Customer text, voice affect, image/video/audio evidence, and company-system events."),
        ("02", "Concepts", "Signals become compact task-relevant concepts with source, confidence and recency."),
        ("03", "Conflict engine", "Contradictions are preserved instead of silently averaging evidence together."),
        ("04", "Top-K JSpace", "Only K concepts stay active. Smaller K increases selectivity and the risk of dropping useful evidence."),
        ("05", "Agent response", "Gemini reasons over the active state and conversation to choose a useful, customer-friendly next response."),
    ]
    node_html = "".join(f'<div class="j-node"><div class="j-node-num">NODE {n}</div><div class="j-node-name">{name}</div><div class="j-node-desc">{desc}</div></div>' for n, name, desc in nodes)
    st.markdown(f'<div class="j-node-grid">{node_html}</div>', unsafe_allow_html=True)

    st.markdown("### Key controls and signals")
    terms = [
        ("JSpace capacity (K)", "Maximum number of concepts allowed in the active workspace at once. K=3 is highly selective; K=8 keeps more context."),
        ("Customer affect intensity", "How strongly the current emotional signal is expressed, from 0% to 100%. It is not a satisfaction or accuracy score."),
        ("Priority", "A ranking signal used to decide which concepts survive capacity filtering. It combines relevance, confidence, conflict importance and recency."),
        ("Recommended next move", "A coaching cue for the support agent: the next action most likely to advance the case without repeating work."),
        ("Evidence & provenance", "Shows where each active concept came from — text, audio, image/video, backend systems, or derived reasoning."),
        ("Researcher view", "Reveals the simulated hidden truth so you can understand why a conflict exists. A real customer-facing product would not expose this."),
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



domains = list_domains()
start_tab, scenario_tab, manual_tab = st.tabs(["◎ Start Here", "✦ Scenario Lab", "◈ Manual Multimodal AI"])

with start_tab:
    render_start_here(domains)

with scenario_tab:
    st.markdown("## Live Scenario Simulator")
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
        scenario_k = st.slider("JSpace capacity K", 3, 10, 5, key="scenario_k")
        seed_text = st.text_input("Optional seed", placeholder="blank = new case", key="scenario_seed")
        seed = int(seed_text) if seed_text.strip().isdigit() else None

    if st.button("Generate scenario", type="primary", use_container_width=True, key="generate_scenario"):
        scenario = generate_scenario(ScenarioControls(domain=scenario_domain, seed=seed))
        scenario, scenario_provider = enhance_scenario_with_gemini(
            scenario, api_key=GEMINI_API_KEY, model=GEMINI_MODEL, channel=CHANNELS[channel_label]["slug"]
        )
        for step in scenario.steps:
            step.customer_turn.affect_source = "text" if channel_label == "Text Messages" else "audio"
        st.session_state.live_scenario = scenario
        st.session_state.live_state = new_scenario_state(scenario, capacity_k=scenario_k)
        st.session_state.live_next_step = 0
        st.session_state.live_started = False
        st.session_state.live_channel = channel_label
        st.session_state.live_scenario_provider = scenario_provider
        st.session_state.live_animate_from = None
        st.rerun()

    scenario = st.session_state.get("live_scenario")
    state = st.session_state.get("live_state")
    next_step = st.session_state.get("live_next_step", 0)
    live_channel = st.session_state.get("live_channel", channel_label)

    if not scenario or not state:
        st.info("Generate a scenario first. The system will prepare a customer profile and problem, but the conversation will not start until you press Start conversation.")
    else:
        st.markdown(
            f'''<div class="j-card j-case"><div class="j-card-title">CASE BRIEF · {html.escape(display_domain(scenario.domain))}</div><div class="j-card-value">{html.escape(scenario.title)}</div><div class="j-card-meta">{html.escape(scenario.problem_summary or scenario.steps[0].customer_turn.text)} · Scenario source: {html.escape(st.session_state.get("live_scenario_provider", "Curated scenario"))}</div></div>''',
            unsafe_allow_html=True,
        )
        render_profile(scenario.customer_profile)

        if not st.session_state.get("live_started", False):
            st.markdown("The customer is ready. Start when you want the first message/call turn to arrive.")
            if st.button("Start conversation ▶", type="primary", use_container_width=True, key="start_live"):
                apply_scenario_step(scenario, state, 0, responder=make_responder(live_channel))
                st.session_state.live_started = True
                st.session_state.live_next_step = 1
                st.session_state.live_animate_from = max(0, len(state.transcript) - 2)
                st.session_state.live_state = state
                st.rerun()
        else:
            chat_col, workspace_col = st.columns([1.12, .88], gap="large")
            with chat_col:
                animate_from = st.session_state.get("live_animate_from")
                render_conversation(state.transcript, live_channel, animate_from=animate_from)
                st.session_state.live_animate_from = None
                if next_step < len(scenario.steps):
                    if st.button("Continue conversation →", type="primary", use_container_width=True, key="continue_scenario"):
                        apply_scenario_step(scenario, state, next_step, responder=make_responder(live_channel))
                        st.session_state.live_next_step = next_step + 1
                        st.session_state.live_animate_from = max(0, len(state.transcript) - 2)
                        st.session_state.live_state = state
                        st.rerun()
                else:
                    st.success("Conversation complete. Generate another case to explore a different customer, domain, media pattern and emotional trajectory.")
            with workspace_col:
                render_workspace(state)

        with st.expander("Researcher view · scenario ground truth", expanded=True):
            st.write("**Domain:**", display_domain(scenario.domain))
            st.write("**Problem summary:**", scenario.problem_summary)
            st.write("**Random conflict present:**", scenario.expected_conflict)
            st.write("**Hidden ground truth:**", scenario.hidden_ground_truth)
            st.write("**Scenario generated/remixed by Gemini:**", scenario.generated_by_ai)
            st.write("**Seed:**", scenario.seed)
            st.write("**Conversation turns planned:**", len(scenario.steps))

with manual_tab:
    st.markdown("## Manual Multimodal AI")
    if AI_CONNECTED:
        st.success(f"Gemini is connected through {GEMINI_MODEL}. Customer messages and supported media attachments can receive model-generated replies.")
    else:
        st.warning("No GEMINI_API_KEY is configured. The conversation still runs with the local fallback, but responses will be more repetitive until Gemini is enabled.")

    m1, m2, m3 = st.columns([1.25, 1, 1])
    with m1:
        manual_domain_label = st.selectbox("Domain", [display_domain(d) for d in domains], key="manual_domain")
        manual_domain = manual_domain_label.lower().replace(" ", "_")
        st.caption(DOMAIN_DESCRIPTIONS.get(manual_domain, ""))
    with m2:
        manual_channel = st.selectbox("Channel", list(CHANNELS), index=3, key="manual_channel")
        st.caption(CHANNELS[manual_channel]["hint"])
    with m3:
        manual_k = st.slider("JSpace capacity K", 3, 10, 5, key="manual_k")
        start_manual = st.button("Start / reset session", type="primary", use_container_width=True)

    if start_manual:
        profile, backend_events = generate_manual_context(manual_domain)
        st.session_state.manual_state_v05 = new_manual_state(capacity_k=manual_k, backend_events=backend_events)
        st.session_state.manual_profile_v05 = profile
        st.session_state.manual_domain_v05 = manual_domain
        st.session_state.manual_channel_v05 = manual_channel
        st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
        st.session_state.manual_animate_from = None
        st.rerun()

    manual_state = st.session_state.get("manual_state_v05")
    manual_profile = st.session_state.get("manual_profile_v05")
    active_manual_domain = st.session_state.get("manual_domain_v05", manual_domain)
    active_manual_channel = st.session_state.get("manual_channel_v05", manual_channel)

    if manual_state and manual_profile:
        st.markdown(
            f'''<div class="j-card j-case"><div class="j-card-title">PRACTICE CASE · {html.escape(display_domain(active_manual_domain))}</div><div class="j-card-value">You are the customer. Explain the problem in your own words.</div><div class="j-card-meta">The company record has been simulated automatically. The support agent should use JSpace to avoid repetition, reconcile conflicts and move you toward a resolution.</div></div>''',
            unsafe_allow_html=True,
        )
        render_profile(manual_profile)

        chat_col, workspace_col = st.columns([1.12, .88], gap="large")
        with chat_col:
            animate_from = st.session_state.get("manual_animate_from")
            render_conversation(manual_state.transcript, active_manual_channel, animate_from=animate_from)
            st.session_state.manual_animate_from = None
        with workspace_col:
            render_workspace(manual_state)

        media_types = {
            "Text Messages": ["png", "jpg", "jpeg", "webp"],
            "Voice Call": ["mp3", "wav", "m4a", "ogg"],
            "Video + Voice": ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "mp4", "mov", "webm"],
            "Multimodal Mix": ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "mp4", "mov", "webm"],
        }
        media_files = st.file_uploader(
            "Attach customer evidence for the next turn (optional)",
            type=media_types[active_manual_channel],
            accept_multiple_files=True,
            key=f"manual_media_{st.session_state.get('manual_media_key', 0)}",
            help="Gemini 3.7 Flash can inspect image, audio and video inputs. Keep uploads small for a responsive free-tier demo.",
        )
        if media_files:
            st.caption("Next-turn media: " + ", ".join(f.name for f in media_files))

        placeholder = "Type the customer's text message…" if active_manual_channel == "Text Messages" else "Type what the customer says on the call…"
        prompt = st.chat_input(placeholder, key="manual_chat_input")
        if prompt:
            media, media_display = read_uploaded_media(media_files)
            media_concepts = analyze_media_for_jspace(
                media, api_key=GEMINI_API_KEY, model=GEMINI_MODEL, domain=active_manual_domain
            )
            response_fn = make_responder(active_manual_channel, media=media)
            manual_customer_turn(
                manual_state, prompt, profile=manual_profile, domain=active_manual_domain,
                responder=response_fn, media_concepts=media_concepts, attachments=media_display,
            )
            st.session_state.manual_state_v05 = manual_state
            st.session_state.manual_animate_from = max(0, len(manual_state.transcript) - 2)
            st.session_state.manual_media_key = st.session_state.get("manual_media_key", 0) + 1
            st.rerun()

        with st.expander("Researcher view · simulated company context", expanded=True):
            st.write("**Domain:**", display_domain(active_manual_domain))
            st.write("**Channel:**", active_manual_channel)
            st.write("**Current company-system events:**", manual_state.backend_history)
            st.write("**Gemini connected:**", AI_CONNECTED)
    else:
        st.info("Choose a domain/channel and start a session. For the fullest multimodal demo, select Multimodal Mix and connect Gemini.")

st.markdown("---")
st.caption("Research MVP · Gemini 3.7 Flash + capacity-limited, conflict-aware shared representations for multimodal customer service.")
