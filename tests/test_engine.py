from backend.app.engine import extract_from_backend, extract_from_turn, merge_concepts, refresh_state
from backend.app.schemas import BackendEvent, CustomerTurn, SessionConfig, SessionState


def test_conflict_is_preserved_inside_capacity():
    state = SessionState(session_id="t", config=SessionConfig(capacity_k=3, preserve_conflicts=True))
    merge_concepts(state.concepts, extract_from_turn(CustomerTurn(
        text="My payment keeps failing. I've tried it three times and it's still not working.",
        audio_tone="frustrated",
    )))
    merge_concepts(state.concepts, extract_from_backend(BackendEvent(
        event_type="payment_status", value="failed"
    )))
    merge_concepts(state.concepts, extract_from_turn(CustomerTurn(
        text="Okay, I think it worked now. It's fine.", audio_tone="frustrated"
    )))
    state = refresh_state(state)
    assert state.conflicts
    active_names = {c.name for c in state.active_concepts}
    assert "authoritative_status" in active_names
    assert "customer_belief_status" in active_names
    assert state.recommended_action_code == "resolve_authoritative_conflict"
