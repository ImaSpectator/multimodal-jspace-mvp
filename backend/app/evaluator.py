from __future__ import annotations

from .schemas import EvaluationResult, GeneratedScenario, SessionState


def evaluate_scenario(scenario: GeneratedScenario, state: SessionState) -> EvaluationResult:
    active_names = {c.name for c in state.active_concepts}
    critical = set(scenario.critical_concepts)
    retained = len(critical & active_names) / len(critical) if critical else 1.0
    conflict_detected = bool(state.conflicts)
    conflict_correct = conflict_detected == scenario.expected_conflict
    action_correct = state.recommended_action_code == scenario.expected_action_code

    # Weighted to reflect the paper/project emphasis: correct action + critical evidence + conflict handling.
    score = 100.0 * (0.45 * float(action_correct) + 0.35 * retained + 0.20 * float(conflict_correct))
    notes: list[str] = []
    if not action_correct:
        notes.append(f"Expected action {scenario.expected_action_code}, got {state.recommended_action_code}.")
    if retained < 1.0:
        missing = sorted(critical - active_names)
        notes.append(f"Critical concepts missing from final JSpace: {', '.join(missing)}.")
    if not conflict_correct:
        notes.append(f"Expected conflict={scenario.expected_conflict}, detected={conflict_detected}.")
    if not notes:
        notes.append("Final JSpace retained the critical evidence and selected the expected next action.")

    return EvaluationResult(
        action_correct=action_correct,
        conflict_expected=scenario.expected_conflict,
        conflict_detected=conflict_detected,
        conflict_correct=conflict_correct,
        critical_evidence_retention=round(retained, 3),
        final_active_concepts=sorted(active_names),
        expected_action_code=scenario.expected_action_code,
        actual_action_code=state.recommended_action_code,
        score=round(score, 1),
        notes=notes,
    )
