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
from .schemas import BackendEvent, CustomerTurn, GeneratedScenario, SessionConfig, SessionState

Responder = Callable[[SessionState, dict, str], tuple[str, str]]


def new_scenario_state(scenario: GeneratedScenario, *, capacity_k: int = 5) -> SessionState:
    return refresh_state(SessionState(
        session_id=f"scenario_{scenario.scenario_id}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=True),
    ))


def apply_scenario_step(
    scenario: GeneratedScenario,
    state: SessionState,
    step_index: int,
    *,
    responder: Responder | None = None,
) -> SessionState:
    if step_index < 0 or step_index >= len(scenario.steps):
        raise IndexError("scenario step out of range")

    step = scenario.steps[step_index]
    decay_recency(state.concepts)
    turn = step.customer_turn
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
    refresh_state(state)

    if responder:
        reply, provider = responder(state, scenario.customer_profile, scenario.domain)
    else:
        reply, provider = state.last_response or "I can help with that.", "Local simulation"
    state.last_response = reply
    state.transcript.append({"role": "agent", "text": reply, "provider": provider, "step": step.label})
    return state


def run_full_scenario(
    scenario: GeneratedScenario,
    *,
    capacity_k: int = 5,
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
) -> SessionState:
    state = SessionState(
        session_id=f"manual_{uuid4().hex[:10]}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=True),
    )
    for event in backend_events or []:
        state.backend_history.append(event.model_dump())
        merge_concepts(state.concepts, extract_from_backend(event))
    return refresh_state(state)


def manual_customer_turn(
    state: SessionState,
    text: str,
    *,
    profile: dict,
    domain: str,
    responder: Responder | None = None,
) -> SessionState:
    emotion, intensity = infer_text_emotion(text)
    turn = CustomerTurn(text=text, emotion=emotion, emotion_intensity=intensity, nonverbal_cue="inferred from typed message")
    decay_recency(state.concepts)
    state.transcript.append({
        "role": "customer",
        "text": text,
        "emotion": emotion,
        "emotion_intensity": intensity,
        "nonverbal_cue": "text-derived affect",
    })
    merge_concepts(state.concepts, extract_from_turn(turn))
    refresh_state(state)
    if responder:
        reply, provider = responder(state, profile, domain)
    else:
        reply, provider = state.last_response or "I can help with that.", "Local simulation"
    state.last_response = reply
    state.transcript.append({"role": "agent", "text": reply, "provider": provider})
    return state
