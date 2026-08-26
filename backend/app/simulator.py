from __future__ import annotations

from copy import deepcopy

from .engine import decay_recency, extract_from_backend, extract_from_image, extract_from_turn, merge_concepts, refresh_state
from .evaluator import evaluate_scenario
from .schemas import GeneratedScenario, ScenarioRunResult, SessionConfig, SessionState


def run_generated_scenario(scenario: GeneratedScenario, *, capacity_k: int = 5,
                           preserve_conflicts: bool = True) -> ScenarioRunResult:
    state = SessionState(
        session_id=f"auto_{scenario.scenario_id}",
        config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=preserve_conflicts),
    )
    step_states: list[SessionState] = []

    for step in scenario.steps:
        decay_recency(state.concepts)
        if step.customer_turn:
            state.transcript.append({
                "role": "customer",
                "text": step.customer_turn.text,
                "audio_tone": step.customer_turn.audio_tone,
                "step": step.label,
            })
            merge_concepts(state.concepts, extract_from_turn(step.customer_turn))
        for event in step.backend_events:
            state.backend_history.append(event.model_dump())
            merge_concepts(state.concepts, extract_from_backend(event))
        for obs in step.image_observations:
            merge_concepts(state.concepts, extract_from_image(obs))

        refresh_state(state)
        if step.customer_turn:
            state.transcript.append({"role": "agent", "text": state.last_response, "step": step.label})
        step_states.append(deepcopy(state))

    evaluation = evaluate_scenario(scenario, state)
    return ScenarioRunResult(
        scenario=scenario,
        final_state=state,
        step_states=step_states,
        evaluation=evaluation,
    )
