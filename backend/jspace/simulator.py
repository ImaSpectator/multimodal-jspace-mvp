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




def update_customer_relationship(profile: dict, state: SessionState, reply: str, provider: str = "") -> None:
    """Update patience/trust from conversation quality without touching satisfaction.

    Patience starts from the simulated customer profile and never increases: useful
    progress leaves it alone, while repetition, fallbacks, and prolonged unresolved
    conflict consume it. Trust moves
    more gently in either direction. A tiny deterministic jitter keeps sessions from
    feeling mechanically identical while keeping tests reproducible.
    """
    if not profile:
        return
    low = (reply or "").lower()
    provider_low = (provider or "").lower()
    agent_turns = sum(1 for row in state.transcript if row.get("role") == "agent")

    concrete_progress = any(token in low for token in [
        "verified", "confirmed", "found", "identified", "root cause", "fixed", "resolved",
        "updated", "removed", "unlocked", "reissued", "refunded", "activated", "next step",
        "i'll apply", "i will apply", "i've corrected", "已经核实", "已经确认", "查到", "根因",
        "已解决", "已修复", "已更新", "下一步", "我会处理", "已经处理",
    ])
    asks_repeat = any(token in low for token in [
        "try again", "restart again", "reset again", "repeat the", "do that again",
        "再试一次", "再重启", "再重置", "重复刚才",
    ])
    fallback = "fallback" in provider_low or "simulation" in provider_low
    prolonged_conflict = bool(state.conflicts) and agent_turns >= 3 and state.session_phase not in {"resolved", "closing", "ended"}

    patience_loss = 0.0
    if fallback:
        patience_loss += 6.0
    if asks_repeat:
        patience_loss += 7.0
    if prolonged_conflict and not concrete_progress:
        patience_loss += 3.0 + min(3.0, max(0, agent_turns - 3) * 0.7)
    if not concrete_progress and agent_turns >= 4 and state.session_phase == "active":
        patience_loss += 1.5
    profile["patience"] = int(round(min(100.0, float(profile.get("patience", 85)) - patience_loss)))

    trust_delta = 0.0
    if state.session_phase in {"resolved", "closing"} or any(x in low for x in ["confirmed resolved", "issue is resolved", "已经解决", "确认已经解决"]):
        trust_delta += 4.0
    elif concrete_progress:
        trust_delta += 1.8
    if fallback:
        trust_delta -= 3.5
    if asks_repeat:
        trust_delta -= 2.5
    if prolonged_conflict and not concrete_progress:
        trust_delta -= 1.5

    # Small deterministic variation (-0.6, 0, +0.6) based on this exact response.
    jitter = ((sum(ord(ch) for ch in (reply or "")) + agent_turns) % 3 - 1) * 0.6
    if abs(trust_delta) > 0.01:
        trust_delta += jitter
    profile["trust"] = int(round(max(0.0, min(100.0, float(profile.get("trust", 55)) + trust_delta))))


def _customer_requests_resolution(text: str) -> bool:
    """Return True for natural customer authorization/request language.

    This deliberately matches normal speech ("I just need this resolved", "can you
    fix it", "please take care of it") rather than only a small set of scripted
    button-like phrases.  Pure status questions such as "is it resolved?" do not
    count as authorization.
    """
    low = (text or "").lower().strip()
    request_patterns = [
        "please fix", "can you fix", "could you fix", "fix it", "fix this",
        "please resolve", "can you resolve", "get this resolved", "need this resolved",
        "need it resolved", "get it sorted", "sort this out", "take care of it",
        "take care of this", "go ahead", "please proceed", "make that change",
        "make the change", "please handle", "do whatever you need", "do it",
        "请修复", "请解决", "帮我解决", "把这个问题解决", "请处理", "帮我处理",
        "请继续处理", "就按这个做", "请直接处理",
    ]
    return any(pattern in low for pattern in request_patterns)


def _maybe_advance_manual_resolution(state: SessionState, text: str) -> None:
    """Advance the manual simulation using the same cause -> fix -> confirm arc as Scenario Lab.

    Manual mode is synchronous.  A real support action may take a few minutes, but the
    simulation should represent that work as completed between the customer's request
    and the agent's response.  It must not create an endless sequence of "I'm checking"
    turns that can never produce an asynchronous result.
    """
    customer_turns = sum(1 for row in state.transcript if row.get("role") == "customer")

    # After a short discovery exchange, make the hidden diagnostic result available.
    # If the customer explicitly asks for a fix after the issue has been established,
    # the same turn completes the simulated work. This compresses the few minutes a
    # real agent would spend working into one synchronous simulation response.
    has_root = any(c.name == "root_cause" for c in state.concepts)
    requests_resolution = _customer_requests_resolution(text)
    if not has_root and (customer_turns >= 4 or (customer_turns >= 2 and requests_resolution)):
        raw = (state.manual_context or {}).get("root_cause_event")
        if raw:
            try:
                event = BackendEvent.model_validate(raw)
                state.backend_history.append(event.model_dump())
                merge_concepts(state.concepts, extract_from_backend(event))
                refresh_state(state)
                has_root = True
            except Exception:
                pass

    if not has_root or not requests_resolution:
        return

    authoritative = next((c for c in state.concepts if c.name == "authoritative_status"), None)
    if not authoritative or str(authoritative.value).lower() == "resolved":
        return

    event = BackendEvent(
        event_type="manual_resolution",
        value="resolved",
        metadata={
            "concept_name": "authoritative_status",
            "concept_value": "resolved",
            "evidence": "simulated support remediation completed during this turn and the system-of-record confirms resolution",
            "relevance": 1.0,
            "confidence": 0.995,
            "conflict_importance": 0.0,
        },
    )
    state.backend_history.append(event.model_dump())
    merge_concepts(state.concepts, extract_from_backend(event))
    state.session_phase = "resolved"
    refresh_state(state)

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
    profile = profile or {}
    state = SessionState(
        session_id=f"manual_{uuid4().hex[:10]}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=True),
        customer_satisfaction=_initial_satisfaction(profile),
        manual_context=deepcopy(profile.get("_manual_case", {})),
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
    _maybe_advance_manual_resolution(state, text)
    closing_tokens = [
        "no other", "nothing else", "that's all", "thats all", "all i needed", "thank you", "thanks", "have a good day", "bye",
        "没有其他", "没别的", "就这些", "没有别的问题", "谢谢", "再见", "祝你", "都正常了",
    ]
    customer_is_closing = any(token in text.lower() for token in closing_tokens)
    if customer_is_closing and (
        state.session_phase == "resolved"
        or (not state.conflicts and state.customer_satisfaction >= 76)
    ):
        state.session_phase = "closing"
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
