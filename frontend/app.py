import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("JSPACE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Multimodal JSpace MVP", layout="wide")
st.title("Multimodal JSpace — Automated Customer Service MVP")
st.caption("Generate multi-domain cases, run them automatically, inspect the active JSpace, and score the result.")


def api(method, path, **kwargs):
    r = requests.request(method, f"{API}{path}", timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def concept_df(concepts):
    if not concepts:
        return pd.DataFrame()
    return pd.DataFrame([{
        "concept": c["name"],
        "value": c["value"],
        "sources": ", ".join(c["sources"]),
        "status": c["status"],
        "confidence": round(c["confidence"], 2),
        "relevance": round(c["task_relevance"], 2),
        "conflict": round(c["conflict_importance"], 2),
    } for c in concepts])


try:
    health = api("GET", "/health")
    domains = api("GET", "/scenarios/domains")["domains"]
except Exception as e:
    st.error(f"Backend unavailable at {API}. Start FastAPI first.\n\n{e}")
    st.stop()

st.success(f"Backend connected — v{health.get('version', '?')}")

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
            payload = {
                "controls": {
                    "domain": domain,
                    "difficulty": difficulty,
                    "include_conflict": conflict_value,
                    "seed": seed,
                },
                "capacity_k": capacity_k,
                "preserve_conflicts": preserve,
            }
            st.session_state.auto_result = api("POST", "/scenarios/autorun", json=payload)

        if st.button("Run random batch of 8", use_container_width=True):
            batch = []
            for _ in range(8):
                payload = {
                    "controls": {
                        "domain": "random",
                        "difficulty": difficulty,
                        "include_conflict": conflict_value,
                        "seed": None,
                    },
                    "capacity_k": capacity_k,
                    "preserve_conflicts": preserve,
                }
                batch.append(api("POST", "/scenarios/autorun", json=payload))
            st.session_state.batch_results = batch
            st.session_state.auto_result = batch[-1]

        st.markdown("**Available domains**")
        st.write(", ".join(d.replace("_", " ") for d in domains))

    with results:
        result = st.session_state.get("auto_result")
        if not result:
            st.info("Generate a scenario to see the automated conversation, hidden ground truth, JSpace evolution, and evaluation.")
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
    st.caption("Use this when you want to type your own customer turns or inject backend/image evidence manually.")

    if "manual_session_id" not in st.session_state:
        state = api("POST", "/sessions", json={"capacity_k": 5, "preserve_conflicts": True})
        st.session_state.manual_session_id = state["session_id"]
        st.session_state.manual_state = state

    sid = st.session_state.manual_session_id
    c1, c2 = st.columns([0.45, 0.55])
    with c1:
        utterance = st.text_area("Customer says", placeholder="My package still hasn't arrived...")
        tone = st.selectbox("Audio cue", ["neutral", "calm", "uncertain", "frustrated", "angry"], key="manual_tone")
        if st.button("Send customer turn") and utterance.strip():
            st.session_state.manual_state = api("POST", f"/sessions/{sid}/turn", json={"text": utterance, "audio_tone": tone})
            st.rerun()

        st.markdown("**Structured backend concept**")
        concept_name = st.text_input("Concept name", placeholder="authoritative_status")
        concept_value = st.text_input("Concept value", placeholder="unresolved")
        if st.button("Inject backend concept") and concept_name.strip() and concept_value.strip():
            st.session_state.manual_state = api("POST", f"/sessions/{sid}/backend-event", json={
                "event_type": "custom",
                "value": concept_value,
                "metadata": {
                    "concept_name": concept_name,
                    "concept_value": concept_value,
                    "relevance": 0.95,
                    "confidence": 0.98,
                    "conflict_importance": 0.7 if concept_name == "authoritative_status" else 0.0,
                },
            })
            st.rerun()

        if st.button("Reset manual session"):
            st.session_state.manual_state = api("POST", f"/sessions/{sid}/reset")
            st.rerun()

    with c2:
        state = st.session_state.manual_state
        st.markdown("**Current JSpace**")
        df = concept_df(state.get("active_concepts", []))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("**Response**")
        st.info(state.get("last_response"))
        if state.get("conflicts"):
            for c in state["conflicts"]:
                st.warning(c["description"])
