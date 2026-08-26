from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

# Streamlit Community Cloud executes this file from frontend/. Add the repo root
# so the research engine can be imported directly. No FastAPI/Render service is needed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.engine import (  # noqa: E402
    decay_recency,
    extract_from_backend,
    extract_from_turn,
    merge_concepts,
    refresh_state,
)
from backend.app.scenario_generator import generate_scenario, list_domains  # noqa: E402
from backend.app.schemas import (  # noqa: E402
    BackendEvent,
    CustomerTurn,
    ScenarioControls,
    SessionConfig,
    SessionState,
)
from backend.app.simulator import run_generated_scenario  # noqa: E402

APP_VERSION = "0.3.0-single-service"

st.set_page_config(page_title="Multimodal JSpace MVP", layout="wide")
st.title("Multimodal JSpace — Automated Customer Service MVP")
st.caption(
    "Single-service edition: scenario generation, JSpace reasoning, conflict detection, "
    "evaluation, and UI all run inside Streamlit."
)
st.success(f"JSpace engine loaded locally — v{APP_VERSION}. No Render/FastAPI backend required.")


def as_dict(model_or_dict):
    if isinstance(model_or_dict, dict):
        return model_or_dict
    if hasattr(model_or_dict, "model_dump"):
        return model_or_dict.model_dump()
    return model_or_dict


def concept_df(concepts):
    if not concepts:
        return pd.DataFrame()
    rows = []
    for raw in concepts:
        c = as_dict(raw)
        rows.append({
            "concept": c["name"],
            "value": c["value"],
            "sources": ", ".join(c["sources"]),
            "status": c["status"],
            "confidence": round(c["confidence"], 2),
            "relevance": round(c["task_relevance"], 2),
            "conflict": round(c["conflict_importance"], 2),
        })
    return pd.DataFrame(rows)


def run_case(domain: str, difficulty: str, conflict_value, seed, capacity_k: int, preserve: bool):
    scenario = generate_scenario(ScenarioControls(
        domain=domain,
        difficulty=difficulty,
        include_conflict=conflict_value,
        seed=seed,
    ))
    return run_generated_scenario(
        scenario,
        capacity_k=capacity_k,
        preserve_conflicts=preserve,
    ).model_dump()


def new_manual_state(capacity_k: int = 5, preserve_conflicts: bool = True) -> SessionState:
    state = SessionState(
        session_id=f"manual_{uuid4().hex[:10]}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=preserve_conflicts),
    )
    return refresh_state(state)


def manual_add_turn(state: SessionState, text: str, tone: str) -> SessionState:
    decay_recency(state.concepts)
    turn = CustomerTurn(text=text, audio_tone=tone)
    state.transcript.append({"role": "customer", "text": text, "audio_tone": tone})
    merge_concepts(state.concepts, extract_from_turn(turn))
    refresh_state(state)
    state.transcript.append({"role": "agent", "text": state.last_response})
    return state


def manual_add_backend_concept(state: SessionState, name: str, value: str) -> SessionState:
    decay_recency(state.concepts)
    event = BackendEvent(
        event_type="custom",
        value=value,
        metadata={
            "concept_name": name,
            "concept_value": value,
            "relevance": 0.95,
            "confidence": 0.98,
            "conflict_importance": 0.7 if name == "authoritative_status" else 0.0,
        },
    )
    state.backend_history.append(event.model_dump())
    merge_concepts(state.concepts, extract_from_backend(event))
    return refresh_state(state)


domains = list_domains()
auto_tab, manual_tab = st.tabs(["Automated Scenario Lab", "Manual JSpace Sandbox"])

with auto_tab:
    controls, results = st.columns([0.34, 0.66], gap="large")

    with controls:
        st.subheader("Scenario generator")
        domain_label = st.selectbox("Domain", ["Random"] + [d.replace("_", " ").title() for d in domains])
        domain = "random" if domain_label == "Random" else domain_label.lower().replace(" ", "_")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        conflict_mode = st.selectbox("Cross-modal / cross-source conflict", ["Automatic", "Always", "Never"])
        conflict_value = {"Automatic": None, "Always": True, "Never": False}[conflict_mode]
        capacity_k = st.slider("JSpace capacity K", 2, 15, 5)
        preserve = st.checkbox("Preserve conflict evidence", value=True)
        seed_text = st.text_input("Optional deterministic seed", placeholder="Leave blank for a new scenario")
        seed = int(seed_text) if seed_text.strip().isdigit() else None

        if st.button("Generate + run full scenario", type="primary", use_container_width=True):
            st.session_state.auto_result = run_case(
                domain, difficulty, conflict_value, seed, capacity_k, preserve
            )

        if st.button("Run random batch of 8", use_container_width=True):
            batch = [
                run_case("random", difficulty, conflict_value, None, capacity_k, preserve)
                for _ in range(8)
            ]
            st.session_state.batch_results = batch
            st.session_state.auto_result = batch[-1]

        st.markdown("**Available domains**")
        st.write(", ".join(d.replace("_", " ") for d in domains))

    with results:
        result = st.session_state.get("auto_result")
        if not result:
            st.info(
                "Generate a scenario to see the automated conversation, hidden ground truth, "
                "JSpace evolution, and evaluation."
            )
        else:
            scenario = result["scenario"]
            final = result["final_state"]
            evaluation = result["evaluation"]

            st.subheader(f"{scenario['title']} — {scenario['domain'].replace('_', ' ').title()}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Score", f"{evaluation['score']:.0f}/100")
            m2.metric("Critical evidence retention", f"{evaluation['critical_evidence_retention']*100:.0f}%")
            m3.metric("Action correct", "Yes" if evaluation["action_correct"] else "No")
            m4.metric("Conflict", "Detected" if evaluation["conflict_detected"] else "None")

            with st.expander("Scenario setup + hidden ground truth", expanded=False):
                st.write("**Customer profile**", scenario["customer_profile"])
                st.write("**Hidden ground truth**", scenario["hidden_ground_truth"])
                st.write("**Expected next action**", scenario["expected_action_code"])
                st.write("**Critical concepts**", scenario["critical_concepts"])
                st.caption(f"seed={scenario['seed']} · difficulty={scenario['difficulty']}")

            st.markdown("### Automated conversation")
            for row in final.get("transcript", []):
                with st.chat_message("assistant" if row.get("role") == "agent" else "user"):
                    st.write(row.get("text", ""))
                    if row.get("audio_tone"):
                        st.caption(f"audio cue: {row['audio_tone']}")

            left, right = st.columns(2)
            with left:
                st.markdown(f"### Final JSpace (K={final['config']['capacity_k']})")
                df = concept_df(final.get("active_concepts", []))
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown("**Recommended next action**")
                st.info(f"{final.get('recommended_action_code')} — {final.get('recommended_action')}")
                st.markdown("**Agent response**")
                st.write(final.get("last_response"))

            with right:
                st.markdown("### Conflicts")
                if final.get("conflicts"):
                    for c in final["conflicts"]:
                        st.warning(f"{c['severity'].upper()}: {c['description']}")
                else:
                    st.success("No conflict detected.")

                st.markdown("### Automatic evaluator")
                for note in evaluation.get("notes", []):
                    st.write(f"- {note}")

            st.markdown("### JSpace evolution by step")
            for i, (step, state) in enumerate(zip(scenario["steps"], result["step_states"]), 1):
                with st.expander(f"{i}. {step['label']}"):
                    st.write("**Active concepts**")
                    step_df = concept_df(state.get("active_concepts", []))
                    if not step_df.empty:
                        st.dataframe(step_df, use_container_width=True, hide_index=True)
                    st.write("**Action:**", state.get("recommended_action_code"), "—", state.get("recommended_action"))
                    if state.get("conflicts"):
                        for c in state["conflicts"]:
                            st.warning(c["description"])

    batch = st.session_state.get("batch_results")
    if batch:
        st.divider()
        st.subheader("Batch results")
        batch_df = pd.DataFrame([{
            "domain": r["scenario"]["domain"].replace("_", " "),
            "title": r["scenario"]["title"],
            "difficulty": r["scenario"]["difficulty"],
            "conflict expected": r["scenario"]["expected_conflict"],
            "score": r["evaluation"]["score"],
            "evidence retention": r["evaluation"]["critical_evidence_retention"],
            "action correct": r["evaluation"]["action_correct"],
        } for r in batch])
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.metric("Batch average score", f"{batch_df['score'].mean():.1f}/100")

with manual_tab:
    st.subheader("Manual session")
    st.caption("Type customer turns or inject structured backend evidence manually.")

    if "manual_state_v03" not in st.session_state:
        st.session_state.manual_state_v03 = new_manual_state()

    state: SessionState = st.session_state.manual_state_v03

    # Manual mode gets its own capacity controls so it is useful for experimentation.
    mc1, mc2 = st.columns(2)
    with mc1:
        manual_k = st.slider("Manual JSpace K", 2, 15, state.config.capacity_k, key="manual_k")
    with mc2:
        manual_preserve = st.checkbox(
            "Manual: preserve conflict evidence", value=state.config.preserve_conflicts, key="manual_preserve"
        )
    if manual_k != state.config.capacity_k or manual_preserve != state.config.preserve_conflicts:
        state.config = SessionConfig(capacity_k=manual_k, preserve_conflicts=manual_preserve)
        st.session_state.manual_state_v03 = refresh_state(state)
        state = st.session_state.manual_state_v03

    c1, c2 = st.columns([0.45, 0.55])
    with c1:
        utterance = st.text_area("Customer says", placeholder="My package still hasn't arrived...")
        tone = st.selectbox(
            "Audio cue", ["neutral", "calm", "uncertain", "frustrated", "angry"], key="manual_tone"
        )
        if st.button("Send customer turn") and utterance.strip():
            st.session_state.manual_state_v03 = manual_add_turn(state, utterance, tone)
            st.rerun()

        st.markdown("**Structured backend concept**")
        concept_name = st.text_input("Concept name", placeholder="authoritative_status")
        concept_value = st.text_input("Concept value", placeholder="unresolved")
        if st.button("Inject backend concept") and concept_name.strip() and concept_value.strip():
            st.session_state.manual_state_v03 = manual_add_backend_concept(
                state, concept_name.strip(), concept_value.strip()
            )
            st.rerun()

        if st.button("Reset manual session"):
            st.session_state.manual_state_v03 = new_manual_state(manual_k, manual_preserve)
            st.rerun()

    with c2:
        state = st.session_state.manual_state_v03
        st.markdown("**Current JSpace**")
        df = concept_df(state.active_concepts)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No active concepts yet.")
        st.markdown("**Response**")
        st.info(state.last_response or "Waiting for customer input.")
        if state.conflicts:
            for c in state.conflicts:
                st.warning(c.description)
