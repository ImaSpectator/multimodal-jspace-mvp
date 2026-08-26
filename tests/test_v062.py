from pathlib import Path
import sys
import types

from backend.jspace_v062 import RUNTIME_VERSION
from backend.jspace_v062.ai_provider import _cached_client, stream_support_reply
from backend.jspace_v062.scenario_generator import generate_scenario, list_domains
from backend.jspace_v062.schemas import ScenarioControls
from backend.jspace_v062.simulator import end_scenario_session, new_scenario_state


def _install_streaming_google(monkeypatch):
    calls = {"client": 0, "stream": 0, "timeouts": []}

    class FakeChunk:
        def __init__(self, text): self.text = text

    class FakeModels:
        def generate_content_stream(self, model, contents, config=None):
            calls["stream"] += 1
            yield FakeChunk("I can check that ")
            yield FakeChunk("for you now.")
        def generate_content(self, model, contents, config=None):
            return FakeChunk("OK")

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            calls["client"] += 1
            calls["timeouts"].append(getattr(http_options, "timeout", None))
            self.models = FakeModels()

    class FakeThinkingConfig:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class FakeHttpOptions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type): return {"data": data, "mime_type": mime_type}

    fake_types = types.ModuleType("google.genai.types")
    fake_types.ThinkingConfig = FakeThinkingConfig
    fake_types.GenerateContentConfig = FakeGenerateContentConfig
    fake_types.HttpOptions = FakeHttpOptions
    fake_types.Part = FakePart
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    _cached_client.cache_clear()
    return calls


def test_runtime_version():
    assert RUNTIME_VERSION == "0.6.2"


def test_streaming_reply_uses_bounded_timeout_and_chunks(monkeypatch):
    calls = _install_streaming_google(monkeypatch)
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=6202))
    state = new_scenario_state(scenario, capacity_k=4)
    chunks = list(stream_support_reply(
        state, scenario.customer_profile, scenario.domain,
        api_key="v062-test", timeout_ms=8000, max_attempts=2,
    ))
    assert calls["client"] == 1
    assert calls["timeouts"] == [8000]
    assert calls["stream"] == 1
    assert chunks[-1][2] is True
    assert chunks[-1][0] == "I can check that for you now."


def test_explicit_scenario_end_does_not_claim_resolution():
    scenario = generate_scenario(ScenarioControls(domain="delivery", seed=6203))
    state = new_scenario_state(scenario, capacity_k=4)
    end_scenario_session(state)
    assert state.session_ended
    assert state.session_phase == "ended"
    assert "practice session" in state.transcript[-1]["text"].lower()
    assert "resolved" not in state.transcript[-1]["text"].lower()


def test_all_domains_still_generate():
    for domain in list_domains():
        scenario = generate_scenario(ScenarioControls(domain=domain, seed=6204))
        assert scenario.domain == domain
        assert len(scenario.steps) >= 7


def test_frontend_v062_speed_and_tab_controls_present():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    required = [
        "0.6.2-fast-stream", "stream_support_reply", "Support Agent is typing",
        "AI response profile", "Test Gemini connection", "on_change=_on_main_tab_change",
        "scenario-live-anchor", "End session", "Email link", "settings_auto_scroll",
        "timeout_ms", "generation_epoch",
    ]
    for text in required:
        assert text in source


def test_streamlit_requirement_supports_dynamic_tabs():
    req = Path("frontend/requirements.txt").read_text(encoding="utf-8")
    assert "streamlit>=1.62" in req
