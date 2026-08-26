from backend.jspace_v061 import RUNTIME_VERSION
from backend.jspace_v061.scenario_generator import generate_scenario, list_domains
from backend.jspace_v061.schemas import ScenarioControls
from backend.jspace_v061.simulator import (
    append_agent_reply,
    apply_manual_customer_message,
    apply_scenario_customer_step,
    end_manual_session,
    new_manual_state,
    new_scenario_state,
)


def test_versioned_runtime_imports_are_atomic():
    assert RUNTIME_VERSION == "0.6.1"
    assert callable(apply_scenario_customer_step)
    assert callable(end_manual_session)


def test_versioned_runtime_runs_all_domains():
    for domain in list_domains():
        scenario = generate_scenario(ScenarioControls(domain=domain, seed=6101))
        state = new_scenario_state(scenario, capacity_k=4)
        apply_scenario_customer_step(scenario, state, 0)
        append_agent_reply(state, "I understand. Let me verify the current state and take the next step.", "test")
        assert state.transcript[0]["role"] == "customer"
        assert state.transcript[1]["role"] == "agent"
        assert len(state.active_concepts) <= 4


def test_versioned_manual_session_flow():
    state = new_manual_state(capacity_k=4, profile={"trust": 60, "relationship": "neutral"})
    apply_manual_customer_message(state, "I still need help with this issue.")
    append_agent_reply(state, "I can help. I’ll verify the current state before we continue.", "test")
    end_manual_session(state)
    assert state.session_ended
