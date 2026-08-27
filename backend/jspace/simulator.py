from __future__ import annotations

from copy import deepcopy
from typing import Callable
from uuid import uuid4

from .engine import (
    decay_recency,
    extract_from_backend,
    extract_from_image,
    extract_from_turn,
    infer_text_emotion,
    merge_concepts,
    refresh_state,
)
from .schemas import BackendEvent, Concept, CustomerTurn, GeneratedScenario, SessionConfig, SessionState

Responder = Callable[[SessionState, dict, str], tuple[str, str]]


def _initial_satisfaction(profile: dict) -> float:
    relationship = str(profile.get("relationship", "neutral")).lower()
    trust = float(profile.get("trust", 55))
    base = 48 + (trust - 50) * 0.22
    base += {"loyal": 8, "positive": 5, "neutral": 0, "strained": -9, "at risk": -15}.get(relationship, 0)
    return round(max(18, min(82, base)), 1)


def new_scenario_state(scenario: GeneratedScenario, *, capacity_k: int = 4) -> SessionState:
    return refresh_state(SessionState(
        session_id=f"scenario_{scenario.scenario_id}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=True),
        customer_satisfaction=_initial_satisfaction(scenario.customer_profile),
    ))


def _phase_for_label(label: str) -> str:
    low = label.lower()
    if "no other concerns" in low or "customer closes" in low:
        return "closing"
    if "resolution confirmed" in low or "issue resolved" in low:
        return "resolved"
    if "resolution" in low or "remediation" in low:
        return "resolving"
    return "active"


def _update_satisfaction(state: SessionState, reply: str) -> None:
    """Lightweight experience score, not an accuracy score.

    It reflects whether the interaction is likely to feel useful: progress, empathy,
    no repetition, and actual resolution improve it; unresolved conflict and generic
    fallback language reduce it.
    """
    low = reply.lower()
    delta = 0.0
    if any(x in low for x in ["i can see", "i understand", "thanks for", "appreciate"]):
        delta += 2.5
    if any(x in low for x in ["i'll", "i will", "let me", "next step", "confirmed", "fixed", "resolved"]):
        delta += 4.0
    if any(x in low for x in ["won't ask you to repeat", "won't make you repeat", "already tried"]):
        delta += 2.0
    if state.session_phase == "resolved":
        delta += 10.0
    elif state.session_phase == "closing":
        delta += 5.0
    if state.conflicts and state.session_phase not in {"resolved", "closing"}:
        delta -= 3.0
    if state.current_emotion in {"angry", "distressed", "frustrated", "impatient"}:
        delta -= 1.5
    if "local fallback" in low or "i can help with that" == low.strip():
        delta -= 4.0
    state.customer_satisfaction = round(max(0.0, min(100.0, state.customer_satisfaction + delta)), 1)


def apply_scenario_customer_step(scenario: GeneratedScenario, state: SessionState, step_index: int) -> SessionState:
    """Apply only the customer/evidence half of a scenario turn.

    This lets the UI display the customer's message immediately while the agent is
    still generating its response.
    """
    if step_index < 0 or step_index >= len(scenario.steps):
        raise IndexError("scenario step out of range")

    step = scenario.steps[step_index]
    # Idempotence guard: Streamlit reruns or a double-click must not append the same
    # scripted customer step twice while an AI response is still being generated.
    if any(row.get("role") == "customer" and row.get("step") == step.label for row in state.transcript):
        return refresh_state(state)
    decay_recency(state.concepts)
    turn = step.customer_turn
    # Let the simulated customer's future tone react to how the support experience is going.
    if state.session_phase == "active" and state.customer_satisfaction >= 76 and turn.emotion in {"angry", "frustrated", "impatient", "skeptical", "disappointed"}:
        turn = turn.model_copy(update={"emotion": "hopeful", "emotion_intensity": min(0.55, turn.emotion_intensity)})
    elif state.session_phase == "active" and state.customer_satisfaction <= 30 and turn.emotion in {"neutral", "curious", "hopeful"}:
        turn = turn.model_copy(update={"emotion": "frustrated", "emotion_intensity": max(0.68, turn.emotion_intensity)})
    state.transcript.append({
        "role": "customer",
        "text": turn.text,
        "emotion": turn.emotion,
        "emotion_intensity": turn.emotion_intensity,
        "nonverbal_cue": turn.nonverbal_cue,
        "step": step.label,
    })
    merge_concepts(state.concepts, extract_from_turn(turn))
    for event in step.backend_events:
        state.backend_history.append(event.model_dump())
        merge_concepts(state.concepts, extract_from_backend(event))
    for obs in step.image_observations:
        merge_concepts(state.concepts, extract_from_image(obs))

    state.session_phase = _phase_for_label(step.label)  # type: ignore[assignment]
    return refresh_state(state)


def append_agent_reply(state: SessionState, reply: str, provider: str, *, step_label: str | None = None) -> SessionState:
    # Avoid duplicate appends if a completed Streamlit action is replayed during a rerun.
    if state.transcript:
        last = state.transcript[-1]
        if last.get("role") == "agent" and str(last.get("text", "")).strip() == reply.strip() and last.get("step") == step_label:
            state.last_response = reply
            return state
    state.last_response = reply
    state.transcript.append({"role": "agent", "text": reply, "provider": provider, "step": step_label})
    _update_satisfaction(state, reply)
    if state.session_phase == "closing":
        state.session_phase = "ended"
        state.session_ended = True
    return state


def apply_scenario_step(
    scenario: GeneratedScenario,
    state: SessionState,
    step_index: int,
    *,
    responder: Responder | None = None,
) -> SessionState:
    """Compatibility helper used by tests and non-interactive runs."""
    apply_scenario_customer_step(scenario, state, step_index)
    step = scenario.steps[step_index]
    if responder:
        reply, provider = responder(state, scenario.customer_profile, scenario.domain)
    else:
        reply, provider = state.last_response or "I can help with that.", "Local simulation"
    return append_agent_reply(state, reply, provider, step_label=step.label)


def run_full_scenario(
    scenario: GeneratedScenario,
    *,
    capacity_k: int = 4,
    responder: Responder | None = None,
) -> tuple[SessionState, list[SessionState]]:
    state = new_scenario_state(scenario, capacity_k=capacity_k)
    snapshots: list[SessionState] = []
    for i in range(len(scenario.steps)):
        apply_scenario_step(scenario, state, i, responder=responder)
        snapshots.append(deepcopy(state))
    return state, snapshots


def new_manual_state(
    *,
    capacity_k: int,
    backend_events: list[BackendEvent] | None = None,
    profile: dict | None = None,
) -> SessionState:
    state = SessionState(
        session_id=f"manual_{uuid4().hex[:10]}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=True),
        customer_satisfaction=_initial_satisfaction(profile or {}),
    )
    for event in backend_events or []:
        state.backend_history.append(event.model_dump())
        merge_concepts(state.concepts, extract_from_backend(event))
    return refresh_state(state)


def apply_manual_customer_message(
    state: SessionState,
    text: str,
    *,
    media_concepts: list[Concept] | None = None,
    attachments: list[dict] | None = None,
    affect_source: str = "text",
) -> SessionState:
    emotion, intensity = infer_text_emotion(text)
    source = affect_source if affect_source in {"text", "audio", "video"} else "text"
    turn = CustomerTurn(
        text=text,
        emotion=emotion,
        emotion_intensity=intensity,
        nonverbal_cue=f"inferred from {source} turn",
        affect_source=source,
    )
    decay_recency(state.concepts)
    state.transcript.append({
        "role": "customer",
        "text": text,
        "emotion": emotion,
        "emotion_intensity": intensity,
        "nonverbal_cue": f"{source}-derived affect",
        "attachments": attachments or [],
    })
    merge_concepts(state.concepts, extract_from_turn(turn))
    if media_concepts:
        merge_concepts(state.concepts, media_concepts)
    return refresh_state(state)


def manual_customer_turn(
    state: SessionState,
    text: str,
    *,
    profile: dict,
    domain: str,
    responder: Responder | None = None,
    media_concepts: list[Concept] | None = None,
    attachments: list[dict] | None = None,
    affect_source: str = "text",
) -> SessionState:
    """Compatibility helper that performs both halves of a manual turn."""
    apply_manual_customer_message(
        state, text, media_concepts=media_concepts, attachments=attachments, affect_source=affect_source
    )
    if responder:
        reply, provider = responder(state, profile, domain)
    else:
        reply, provider = state.last_response or "I can help with that.", "Local simulation"
    return append_agent_reply(state, reply, provider)


def end_manual_session(state: SessionState) -> SessionState:
    if state.session_ended:
        return state
    closing = "Thanks for contacting support. I hope the next steps are clear, and I hope you have a good day."
    state.session_phase = "ended"
    state.session_ended = True
    state.last_response = closing
    state.transcript.append({"role": "agent", "text": closing, "provider": "Session close"})
    _update_satisfaction(state, closing)
    return state


def end_scenario_session(state: SessionState) -> SessionState:
    """Explicitly end a Scenario Lab practice session without pretending the issue was resolved."""
    if state.session_ended:
        return state
    closing = (
        "Understood — I’ll end this practice session here. I’ll leave the case context as-is and won’t mark it complete. "
        "Thanks for the conversation."
    )
    state.session_phase = "ended"
    state.session_ended = True
    state.last_response = closing
    state.transcript.append({"role": "agent", "text": closing, "provider": "Session close"})
    _update_satisfaction(state, closing)
    return state
