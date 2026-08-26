from pathlib import Path
import sys
import types

from backend.jspace_v063 import RUNTIME_VERSION
from backend.jspace_v063.ai_provider import _cached_client, stream_support_reply
from backend.jspace_v063.scenario_generator import generate_scenario, list_domains
from backend.jspace_v063.schemas import ScenarioControls
from backend.jspace_v063.simulator import end_scenario_session, new_scenario_state


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
    assert RUNTIME_VERSION == "0.6.3"


def test_streaming_reply_uses_bounded_timeout_and_chunks(monkeypatch):
    calls = _install_streaming_google(monkeypatch)
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=6202))
    state = new_scenario_state(scenario, capacity_k=4)
    chunks = list(stream_support_reply(
        state, scenario.customer_profile, scenario.domain,
        api_key="v062-test", timeout_ms=12000, max_attempts=2,
    ))
    assert calls["client"] == 1
    assert calls["timeouts"] == [12000]
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
        "0.6.3-gemini-chat-fix", "stream_support_reply", "Support Agent is typing",
        "AI response profile", "Test Gemini connection", "on_change=_on_main_tab_change",
        "scenario-live-anchor", "End session", "Email link", "settings_auto_scroll",
        "timeout_ms", "generation_epoch",
    ]
    for text in required:
        assert text in source


def test_streamlit_requirement_supports_dynamic_tabs():
    req = Path("frontend/requirements.txt").read_text(encoding="utf-8")
    assert "streamlit>=1.62" in req


def test_frontend_timeout_never_below_gemini_minimum_and_toolbar_precedes_hero():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "timeout_ms, attempts, history = 12000, 2, 4" in source
    assert "scenario_timeout_ms\": 12000" in source
    assert "timeout_ms=12000" in source
    assert "timeout_ms=5000" not in source
    controls = source.index("# Compact utility controls in the top-right.")
    hero = source.index('<div class="j-hero">')
    assert controls < hero


def test_gemini_timeout_is_clamped_to_minimum(monkeypatch):
    calls = _install_streaming_google(monkeypatch)
    _cached_client.cache_clear()
    _cached_client("too-short-test", 5000)
    assert calls["timeouts"] == [10000]


def test_frontend_shows_provider_and_manual_chat_composer_order():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "AI provider · " in source
    assert "Backup responder · " in source
    assert 'key="manual_chat_input"' in source
    assert "Press Enter or click Send message" in source
    composer = source.index('key="manual_chat_input"')
    suggestion = source.index('"Use suggested prompt"', composer)
    assert composer < suggestion
    assert "on_click=_use_manual_suggestion" in source


def test_probe_connection_uses_safe_timeout_in_frontend():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "probe_gemini(api_key=GEMINI_API_KEY, model=GEMINI_MODEL, timeout_ms=12000)" in source
    assert "timeout_ms=5000" not in source
