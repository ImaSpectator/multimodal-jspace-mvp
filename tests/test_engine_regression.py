from backend.jspace_v063.engine import extract_from_turn, merge_concepts, refresh_state
from backend.jspace_v063.schemas import CustomerTurn, SessionConfig, SessionState


def test_capacity_limit_is_respected():
    state = SessionState(session_id="x", config=SessionConfig(capacity_k=3))
    turns = [
        CustomerTurn(text="My payment failed and I already tried three times", emotion="frustrated", emotion_intensity=0.8),
        CustomerTurn(text="I am worried because I need this today", emotion="anxious", emotion_intensity=0.75),
        CustomerTurn(text="Are you sure this is right?", emotion="skeptical", emotion_intensity=0.7),
    ]
    for turn in turns:
        merge_concepts(state.concepts, extract_from_turn(turn))
        refresh_state(state)
    assert len(state.active_concepts) <= 3


def test_emotion_intensity_is_not_fixed():
    low = extract_from_turn(CustomerTurn(text="okay", emotion="frustrated", emotion_intensity=0.35))
    high = extract_from_turn(CustomerTurn(text="this is terrible", emotion="frustrated", emotion_intensity=0.92))
    low_conf = next(c.confidence for c in low if c.name == "customer_emotion")
    high_conf = next(c.confidence for c in high if c.name == "customer_emotion")
    assert high_conf > low_conf
