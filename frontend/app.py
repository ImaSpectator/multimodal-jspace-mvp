from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ai_provider import DEFAULT_MODEL, generate_support_reply  # noqa: E402
from backend.app.scenario_generator import generate_manual_context, generate_scenario, list_domains  # noqa: E402
from backend.app.schemas import ScenarioControls  # noqa: E402
from backend.app.simulator import (  # noqa: E402
    apply_scenario_step,
    manual_customer_turn,
    new_manual_state,
    new_scenario_state,
)

APP_VERSION = "0.4.0-conversational"

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
  --j-cyan: #61F4FF;
  --j-blue: #5B8CFF;
  --j-violet: #B477FF;
  --j-bg: #060A13;
  --j-panel: rgba(15, 23, 42, 0.68);
  --j-border: rgba(112, 225, 255, 0.18);
  --j-text: #E9F3FF;
  --j-muted: #8CA4BE;
}
.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(91,140,255,.16), transparent 31%),
    radial-gradient(circle at 88% 14%, rgba(180,119,255,.12), transparent 28%),
    linear-gradient(180deg, #070B14 0%, #050811 55%, #060912 100%);
  color: var(--j-text);
}
.block-container { max-width: 1480px; padding-top: 1.6rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stToolbar"] { right: 1rem; }
.j-hero {
  padding: 1.5rem 1.7rem;
  border: 1px solid var(--j-border);
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(16,28,52,.84), rgba(10,17,33,.72));
  box-shadow: 0 18px 60px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.03);
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.j-hero:before { content:""; position:absolute; inset:0 0 auto 0; height:2px; background:linear-gradient(90deg,var(--j-cyan),var(--j-violet),transparent); }
.j-kicker { color: var(--j-cyan); letter-spacing:.18em; font-size:.72rem; font-weight:700; }
.j-title { font-size:2.2rem; line-height:1.08; font-weight:720; margin:.35rem 0 .45rem; color:#F6FAFF; }
.j-sub { color:var(--j-muted); max-width:850px; font-size:.98rem; }
.j-card {
  border: 1px solid var(--j-border);
  border-radius: 16px;
  background: var(--j-panel);
  padding: .95rem 1rem;
  margin: .45rem 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
.j-card-title { color:#CFEAFF; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.25rem; }
.j-card-value { color:#F3F8FF; font-size:1.02rem; font-weight:650; overflow-wrap:anywhere; }
.j-card-meta { color:var(--j-muted); font-size:.76rem; margin-top:.3rem; }
.j-concept { border-left:3px solid var(--j-blue); }
.j-concept.disputed { border-left-color:#FFB454; }
.j-concept.unresolved { border-left-color:#FF6D8A; }
.j-conflict { border:1px solid rgba(255,171,82,.35); background:rgba(86,50,18,.23); }
.j-next { border:1px solid rgba(97,244,255,.28); background:linear-gradient(135deg,rgba(16,62,76,.3),rgba(23,34,68,.42)); }
.j-profile-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; margin:.4rem 0 .8rem; }
.j-profile-cell { border:1px solid rgba(126,168,214,.14); background:rgba(11,19,34,.55); border-radius:12px; padding:.62rem .75rem; }
.j-profile-label { color:#7891AD; font-size:.66rem; text-transform:uppercase; letter-spacing:.08em; }
.j-profile-value { color:#ECF5FF; font-size:.9rem; font-weight:600; margin-top:.15rem; }
.j-pill { display:inline-block; padding:.18rem .5rem; border-radius:999px; border:1px solid rgba(97,244,255,.24); background:rgba(97,244,255,.08); color:#BFF9FF; font-size:.72rem; margin-right:.35rem; }
.j-ai-live { border-color:rgba(100,255,184,.28); color:#A9FFD2; background:rgba(54,154,105,.09); }
.j-ai-local { border-color:rgba(255,195,104,.25); color:#FFD7A0; background:rgba(139,88,31,.10); }
[data-testid="stMetric"] { border:1px solid rgba(110,180,255,.12); background:rgba(10,18,33,.45); border-radius:14px; padding:.6rem .8rem; }
[data-testid="stChatMessage"] { border:1px solid rgba(105,155,210,.10); border-radius:16px; background:rgba(8,15,29,.54); margin-bottom:.55rem; }
.stTabs [data-baseweb="tab-list"] { gap:.45rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:.45rem .85rem; background:rgba(12,21,38,.62); }
.stButton > button { border-radius:11px; border:1px solid rgba(97,244,255,.22); }
hr { border-color:rgba(140,175,215,.12) !important; }
@media (max-width: 900px) { .j-profile-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .j-title { font-size:1.65rem; } }
</style>
""",
    unsafe_allow_html=True,
)


def _secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else None
    except Exception:
        return default


OPENAI_API_KEY = _secret("OPENAI_API_KEY")
OPENAI_MODEL = _secret("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
AI_CONNECTED = bool(OPENAI_API_KEY)

st.markdown(
    f"""
<div class="j-hero">
  <div class="j-kicker">JSPACE // LIVE RESEARCH MVP</div>
  <div class="j-title">Multimodal customer-service reasoning, as an evolving workspace.</div>
  <div class="j-sub">Conversations now unfold turn by turn. JSpace keeps a capacity-limited set of active evidence, preserves important conflicts, and adapts to customer emotion and relationship context.</div>
  <div style="margin-top:.8rem">
    <span class="j-pill">v{APP_VERSION}</span>
    <span class="j-pill {'j-ai-live' if AI_CONNECTED else 'j-ai-local'}">{'OpenAI connected · ' + html.escape(OPENAI_MODEL) if AI_CONNECTED else 'Local simulation · add OPENAI_API_KEY for live AI'}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def display_domain(domain: str) -> str:
    return domain.replace("_", " ").title()


def concept_rows(state) -> pd.DataFrame:
    rows = []
    for c in state.active_concepts:
        rows.append({
            "Concept": c.name.replace("_", " ").title(),
            "Value": c.value,
            "Status": c.status,
            "Sources": ", ".join(c.sources),
            "Priority": round(c.score, 2),
            "Confidence": round(c.confidence, 2),
        })
    return pd.DataFrame(rows)


def profile_html(profile: dict) -> str:
    items = [
        ("Customer", profile.get("name", "—")),
        ("Tenure", profile.get("tenure", "—")),
        ("Relationship", profile.get("relationship", "—")),
        ("Loyalty", profile.get("loyalty_tier", "—")),
        ("Contacts · 90d", profile.get("previous_contacts_90d", "—")),
        ("Value segment", profile.get("value_segment", "—")),
        ("Style", profile.get("communication_style", "—")),
        ("Tech comfort", profile.get("tech_comfort", "—")),
    ]
    cells = "".join(
        f'<div class="j-profile-cell"><div class="j-profile-label">{html.escape(str(k))}</div><div class="j-profile-value">{html.escape(str(v))}</div></div>'
        for k, v in items
    )
    return f'<div class="j-profile-grid">{cells}</div>'


def render_profile(profile: dict) -> None:
    st.markdown(profile_html(profile), unsafe_allow_html=True)
    patience, trust = st.columns(2)
    patience.metric("Patience", f"{profile.get('patience', 0)} / 100")
    trust.metric("Trust in company", f"{profile.get('trust', 0)} / 100")


def render_conversation(transcript: list[dict]) -> None:
    for row in transcript:
        role = row.get("role")
        with st.chat_message("assistant" if role == "agent" else "user"):
            st.write(row.get("text", ""))
            if role == "customer":
                emotion = row.get("emotion")
                intensity = row.get("emotion_intensity")
                cue = row.get("nonverbal_cue")
                details = []
                if emotion:
                    details.append(str(emotion).replace("_", " ").title())
                if isinstance(intensity, (int, float)):
                    details.append(f"{intensity:.0%} intensity")
                if cue:
                    details.append(str(cue))
                if details:
                    st.caption(" · ".join(details))
            elif row.get("provider"):
                st.caption(f"Agent engine: {row['provider']}")


def render_workspace(state, profile: dict) -> None:
    st.markdown("#### Live JSpace")
    m1, m2, m3 = st.columns(3)
    m1.metric("Active", f"{len(state.active_concepts)} / {state.config.capacity_k}")
    m2.metric("Conflicts", len(state.conflicts))
    m3.metric("Emotion", (state.current_emotion or "—").title())

    if state.current_emotion:
        st.caption(f"Customer affect intensity · {state.current_emotion_intensity:.0%}")
        st.progress(max(0.0, min(1.0, state.current_emotion_intensity)))

    if not state.active_concepts:
        st.info("The workspace will populate as evidence arrives.")
    else:
        for c in state.active_concepts:
            cls = "j-concept " + c.status
            sources = " · ".join(s.title() for s in c.sources)
            st.markdown(
                f"""
<div class="j-card {cls}">
  <div class="j-card-title">{html.escape(c.name.replace('_',' ').title())}</div>
  <div class="j-card-value">{html.escape(str(c.value))}</div>
  <div class="j-card-meta">{html.escape(sources)} · priority {c.score:.2f} · {c.status}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if state.conflicts:
        st.markdown("#### Signal conflicts")
        for conflict in state.conflicts:
            st.markdown(
                f'<div class="j-card j-conflict"><div class="j-card-title">{html.escape(conflict.severity.upper())} CONFLICT</div><div class="j-card-value">{html.escape(conflict.description)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
<div class="j-card j-next">
  <div class="j-card-title">Recommended next move</div>
  <div class="j-card-value">{html.escape(state.recommended_action or 'Waiting for more evidence')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Evidence & provenance"):
        df = concept_rows(state)
        if df.empty:
            st.caption("No active evidence yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)


def make_responder():
    def responder(state, profile, domain):
        return generate_support_reply(
            state,
            profile,
            domain,
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            fallback=state.last_response,
        )
    return responder


responder = make_responder()
domains = list_domains()
scenario_tab, manual_tab = st.tabs(["✦ Live Scenario Simulator", "◈ Manual AI Conversation"])

with scenario_tab:
    top_left, top_mid, top_right = st.columns([1.35, 1, 1])
    with top_left:
        domain_label = st.selectbox(
            "Scenario domain",
            ["Random"] + [display_domain(d) for d in domains],
            key="scenario_domain",
        )
        scenario_domain = "random" if domain_label == "Random" else domain_label.lower().replace(" ", "_")
    with top_mid:
        scenario_k = st.slider("JSpace capacity", 3, 10, 5, key="scenario_k")
    with top_right:
        seed_text = st.text_input("Optional seed", placeholder="blank = new case", key="scenario_seed")
        seed = int(seed_text) if seed_text.strip().isdigit() else None

    if st.button("Generate new live scenario", type="primary", use_container_width=True):
        scenario = generate_scenario(ScenarioControls(domain=scenario_domain, seed=seed))
        state = new_scenario_state(scenario, capacity_k=scenario_k)
        apply_scenario_step(scenario, state, 0, responder=responder)
        st.session_state.live_scenario = scenario
        st.session_state.live_state = state
        st.session_state.live_next_step = 1
        st.rerun()

    scenario = st.session_state.get("live_scenario")
    state = st.session_state.get("live_state")
    next_step = st.session_state.get("live_next_step", 0)

    if not scenario or not state:
        st.markdown("---")
        st.info("Choose a domain or Random, then generate a scenario. The conversation will unfold one customer turn at a time instead of dumping the entire case at once.")
        st.caption(f"{len(domains)} domains available: " + ", ".join(display_domain(d) for d in domains))
    else:
        st.markdown("---")
        st.markdown(f"### {scenario.title}")
        st.caption(f"{display_domain(scenario.domain)} · turn {min(next_step, len(scenario.steps))} of {len(scenario.steps)}")
        render_profile(scenario.customer_profile)

        chat_col, workspace_col = st.columns([1.18, 0.82], gap="large")
        with chat_col:
            st.markdown("#### Conversation")
            render_conversation(state.transcript)
            if next_step < len(scenario.steps):
                if st.button("Continue conversation →", type="primary", use_container_width=True, key="continue_scenario"):
                    apply_scenario_step(scenario, state, next_step, responder=responder)
                    st.session_state.live_next_step = next_step + 1
                    st.session_state.live_state = state
                    st.rerun()
            else:
                st.success("Conversation complete. Generate another scenario to explore a different customer, domain, emotional trajectory, and evidence pattern.")
                if st.button("Generate another scenario", use_container_width=True, key="another_scenario"):
                    new_scenario = generate_scenario(ScenarioControls(domain=scenario_domain, seed=None))
                    new_state = new_scenario_state(new_scenario, capacity_k=scenario_k)
                    apply_scenario_step(new_scenario, new_state, 0, responder=responder)
                    st.session_state.live_scenario = new_scenario
                    st.session_state.live_state = new_state
                    st.session_state.live_next_step = 1
                    st.rerun()
        with workspace_col:
            render_workspace(state, scenario.customer_profile)

        with st.expander("Researcher view · scenario ground truth"):
            st.write("**Seed:**", scenario.seed)
            st.write("**Random conflict present:**", scenario.expected_conflict)
            st.write("**Hidden ground truth:**", scenario.hidden_ground_truth)
            st.write("**Total concepts observed:**", len(state.concepts))
            st.write("**Conversation turns planned:**", len(scenario.steps))

with manual_tab:
    st.markdown("### Talk to the support agent yourself")
    if AI_CONNECTED:
        st.success(f"Live AI is connected through OpenAI ({OPENAI_MODEL}). Your typed customer messages receive model-generated replies grounded in the current JSpace.")
    else:
        st.warning("No OPENAI_API_KEY is configured in Streamlit secrets yet. The chat still works with the local fallback, but add the secret to enable actual model-generated replies.")

    mleft, mmid, mright = st.columns([1.35, 1, 1])
    with mleft:
        manual_domain_label = st.selectbox(
            "Customer-service domain",
            [display_domain(d) for d in domains],
            key="manual_domain",
        )
        manual_domain = manual_domain_label.lower().replace(" ", "_")
    with mmid:
        manual_k = st.slider("Manual JSpace capacity", 3, 10, 5, key="manual_k")
    with mright:
        st.write("")
        st.write("")
        start_manual = st.button("Start / reset AI session", use_container_width=True)

    if start_manual:
        profile, backend_events = generate_manual_context(manual_domain)
        manual_state = new_manual_state(capacity_k=manual_k, backend_events=backend_events)
        st.session_state.manual_state_v04 = manual_state
        st.session_state.manual_profile_v04 = profile
        st.session_state.manual_domain_v04 = manual_domain
        st.rerun()

    manual_state = st.session_state.get("manual_state_v04")
    manual_profile = st.session_state.get("manual_profile_v04")
    active_manual_domain = st.session_state.get("manual_domain_v04", manual_domain)

    if manual_state and manual_profile:
        render_profile(manual_profile)
        chat_col, workspace_col = st.columns([1.18, 0.82], gap="large")
        with chat_col:
            st.markdown("#### Customer ↔ AI agent")
            if not manual_state.transcript:
                st.caption("You are the customer. Type the first message below. The AI agent will answer using the active JSpace plus the simulated company record for this case.")
            render_conversation(manual_state.transcript)
        with workspace_col:
            render_workspace(manual_state, manual_profile)

        prompt = st.chat_input("Type a customer message…", key="manual_chat_input")
        if prompt:
            manual_customer_turn(
                manual_state,
                prompt,
                profile=manual_profile,
                domain=active_manual_domain,
                responder=responder,
            )
            st.session_state.manual_state_v04 = manual_state
            st.rerun()
    else:
        st.info("Choose a domain and start a session. Raw backend-concept injection has been removed; the simulated company context is created automatically.")

st.markdown("---")
st.caption("Research MVP · Capacity-limited, conflict-aware shared representations for multimodal customer service.")
