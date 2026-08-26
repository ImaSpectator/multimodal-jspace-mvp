from backend.jspace_v063.ai_provider import _fallback_reply
from backend.jspace_v063.scenario_generator import generate_scenario
from backend.jspace_v063.schemas import ScenarioControls
from backend.jspace_v063.simulator import append_agent_reply, apply_scenario_customer_step, new_scenario_state


def test_contextual_fallback_does_not_repeat_previous_agent_reply():
    scenario = generate_scenario(ScenarioControls(domain="hotel_hospitality", seed=6301))
    state = new_scenario_state(scenario, capacity_k=4)
    apply_scenario_customer_step(scenario, state, 0)
    previous = state.last_response or "I can help with that."
    append_agent_reply(state, previous, "Local simulation")
    # Simulate another customer turn so the backup responder is called again.
    apply_scenario_customer_step(scenario, state, 1)
    candidate = _fallback_reply(state, state.last_response)
    assert candidate
    assert candidate != previous


def test_agent_provider_is_stored_on_transcript():
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=6302))
    state = new_scenario_state(scenario, capacity_k=4)
    apply_scenario_customer_step(scenario, state, 0)
    append_agent_reply(state, "I’ll check that now.", "Gemini · gemini-3.7-flash")
    assert state.transcript[-1]["provider"] == "Gemini · gemini-3.7-flash"
