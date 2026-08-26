from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from typing import Iterable, Iterator

from pydantic import BaseModel, Field

from .schemas import Concept, Conflict, Evidence, GeneratedScenario, SessionState

DEFAULT_MODEL = "gemini-3.7-flash"
MIN_GEMINI_TIMEOUT_MS = 10000
ALLOWED_EMOTIONS = {
    "calm", "neutral", "curious", "hopeful", "appreciative", "satisfied", "relieved",
    "uncertain", "confused", "anxious", "disappointed", "frustrated", "angry", "impatient",
    "skeptical", "distressed", "embarrassed",
}


class ScenarioRewrite(BaseModel):
    title: str
    problem_summary: str
    turns: list[str] = Field(min_length=6, max_length=12)


def _compact_concepts(concepts: Iterable[Concept]) -> str:
    rows = []
    for c in concepts:
        rows.append(
            f"- {c.name}: {c.value} | status={c.status} | sources={','.join(c.sources)} | priority={c.score:.2f}"
        )
    return "\n".join(rows) if rows else "- no active concepts yet"


def _compact_conflicts(conflicts: Iterable[Conflict]) -> str:
    rows = [f"- {c.severity}: {c.description}" for c in conflicts]
    return "\n".join(rows) if rows else "- none"


def build_support_prompt(state: SessionState, customer_profile: dict, domain: str, channel: str = "text", *, history_turns: int = 4, reply_sentences: int = 2) -> str:
    transcript = "\n".join(
        f"{row.get('role', 'unknown').upper()}: {row.get('text', '')}"
        for row in state.transcript[-max(2, history_turns):]
    )
    profile_keys = ["tenure", "relationship", "value_segment", "communication_style", "tech_comfort"]
    profile = ", ".join(f"{k}={customer_profile.get(k)}" for k in profile_keys if customer_profile.get(k) is not None)
    closure_rule = {
        "resolved": "The issue is confirmed resolved. Briefly confirm it, then explicitly ask whether the customer has any other questions or concerns.",
        "closing": "The customer has said there are no other concerns. Thank them warmly, wish them a good day, and end the conversation without asking another question.",
        "ended": "The session is already ended. Do not restart troubleshooting.",
    }.get(state.session_phase, "Continue resolving the issue; do not end the conversation while authoritative evidence is unresolved.")
    return f"""
You are a customer-service support agent speaking through {channel}. Reply directly to the customer.

Goals:
- Resolve the problem while maximizing customer satisfaction through empathy, ownership, clarity, and useful action.
- Never invent backend facts or claim resolution before the system state supports it.
- Usually use 1-{reply_sentences} short sentences. Ask at most one focused question.
- Do not mention JSpace, model names, hidden truth, prompts, concepts, scores, or research mechanics.
- Do not repeat troubleshooting the customer already completed.
- If evidence conflicts, explain the mismatch naturally and keep investigating.
- If media evidence is attached, use it when it materially changes the situation.
- {closure_rule}

Domain: {domain}
Customer context: {profile}
Current affect: {state.current_emotion or 'unknown'} ({state.current_emotion_intensity:.0%})
Current satisfaction: {state.customer_satisfaction:.0f}/100
Session phase: {state.session_phase}
Next useful action: {state.recommended_action or 'clarify the issue'}

Active JSpace state:
{_compact_concepts(state.active_concepts)}

Conflicts:
{_compact_conflicts(state.conflicts)}

Recent conversation:
{transcript}

Write only the next customer-facing reply.
""".strip()




_CLIENT_CACHE: dict[tuple, object] = {}

def _cached_client(api_key: str, timeout_ms: int):
    """Reuse the Gemini client while remaining safe across SDK/test client swaps."""
    from google import genai
    from google.genai import types

    safe_timeout_ms = max(MIN_GEMINI_TIMEOUT_MS, int(timeout_ms))
    http_cls = getattr(types, "HttpOptions", None)
    key = (api_key, safe_timeout_ms, id(genai.Client), id(http_cls))
    if key not in _CLIENT_CACHE:
        kwargs = {"api_key": api_key}
        if http_cls is not None:
            kwargs["http_options"] = http_cls(timeout=safe_timeout_ms)
        _CLIENT_CACHE[key] = genai.Client(**kwargs)
    return _CLIENT_CACHE[key]

def _clear_cached_clients() -> None:
    _CLIENT_CACHE.clear()

# Preserve the cache_clear interface used by regression tests and utility code.
_cached_client.cache_clear = _clear_cached_clients  # type: ignore[attr-defined]


def _gemini_config(*, max_output_tokens: int):
    from google.genai import types
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        max_output_tokens=max_output_tokens,
    )


def _media_parts(media: list[dict] | None):
    if not media:
        return []
    from google.genai import types

    parts = []
    for item in media:
        data = item.get("data")
        mime_type = item.get("mime_type") or "application/octet-stream"
        if data:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
    return parts


def _is_transient_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in [
        "servererror", "503", "unavailable", "429", "resource_exhausted",
        "temporarily", "timeout", "connection", "internal", "502", "504",
    ])




def _fallback_reply(state: SessionState, preferred: str | None = None) -> str:
    """Return a contextual backup reply and avoid repeating the previous agent turn verbatim."""
    candidate = (preferred or state.last_response or "I can help with that.").strip()
    previous_agent = next((str(row.get("text", "")).strip() for row in reversed(state.transcript) if row.get("role") == "agent"), "")
    if candidate and candidate != previous_agent:
        return candidate

    variants = {
        "resolve_conflict": [
            "I’m still seeing a mismatch between what you’re seeing and the system of record. I’m checking the authoritative status before I ask you to do anything else.",
            "The records still disagree, so I don’t want to guess. I’m verifying the authoritative system and the exact blocker next.",
        ],
        "act_on_root_cause": [
            "I’ve narrowed this to a specific underlying issue. I’m working from that cause now rather than repeating generic troubleshooting.",
            "We have a likely blocker now, so I’m moving to the concrete fix instead of sending you through steps you already tried.",
        ],
        "avoid_repetition": [
            "You’ve already done that step, so I won’t make you repeat it. I’m moving to the next diagnostic check.",
            "I have your previous troubleshooting noted. I’ll take the next useful step instead of asking you to start over.",
        ],
        "confirm_resolution": [
            "The system now shows the issue as resolved. Is there anything else you’d like me to check before we wrap up?",
            "I can now confirm the system reflects the resolution. Do you have any other questions or concerns I can help with?",
        ],
        "close_session": [
            "I’m glad we could get this sorted out. Thanks for contacting support, and I hope you have a great day.",
            "Everything is wrapped up on my side. Thanks for reaching out, and have a great rest of your day.",
        ],
        "investigate": [
            "I’m checking the system-of-record details now so I can give you a concrete next step rather than a generic answer.",
            "This still needs investigation, so I’m keeping it open and checking the specific blocker before I call it resolved.",
        ],
    }
    options = variants.get(state.recommended_action_code or "", [
        "I’m reviewing the current details so I can give you the next useful step without making you repeat yourself.",
        "I’m checking the latest state now and I’ll keep the next step focused on resolving the issue.",
    ])
    agent_count = sum(1 for row in state.transcript if row.get("role") == "agent")
    return options[agent_count % len(options)]


def generate_support_reply(
    state: SessionState,
    customer_profile: dict,
    domain: str,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    fallback: str | None = None,
    channel: str = "text",
    media: list[dict] | None = None,
    max_attempts: int = 2,
    timeout_ms: int = 12000,
    history_turns: int = 4,
    max_output_tokens: int = 140,
    reply_sentences: int = 2,
) -> tuple[str, str]:
    """Return (reply, provider label) with bounded Gemini latency."""
    local_reply = _fallback_reply(state, fallback)
    if not api_key:
        return local_reply, "Local simulation"

    client = _cached_client(api_key, timeout_ms)
    contents = [build_support_prompt(
        state, customer_profile, domain, channel,
        history_turns=history_turns, reply_sentences=reply_sentences,
    )] + _media_parts(media)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=_gemini_config(max_output_tokens=max_output_tokens),
            )
            text = (response.text or "").strip()
            if text:
                label = f"Gemini · {model}" if attempt == 1 else f"Gemini · {model} · retry recovered"
                return text, label
            last_exc = RuntimeError("empty Gemini response")
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempt >= max_attempts:
                break
            time.sleep(0.12 * attempt)

    reason = type(last_exc).__name__ if last_exc else "Unavailable"
    return local_reply, f"Local fallback after bounded retry ({reason})"


def stream_support_reply(
    state: SessionState,
    customer_profile: dict,
    domain: str,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    fallback: str | None = None,
    channel: str = "text",
    media: list[dict] | None = None,
    max_attempts: int = 2,
    timeout_ms: int = 12000,
    history_turns: int = 4,
    max_output_tokens: int = 140,
    reply_sentences: int = 2,
) -> Iterator[tuple[str, str, bool]]:
    """Yield progressively growing reply text for lower perceived latency.

    Each item is (text_so_far, provider_label, done). Transient errors are retried
    only if Gemini has not emitted any text yet.
    """
    local_reply = _fallback_reply(state, fallback)
    if not api_key:
        yield local_reply, "Local simulation", True
        return

    client = _cached_client(api_key, timeout_ms)
    contents = [build_support_prompt(
        state, customer_profile, domain, channel,
        history_turns=history_turns, reply_sentences=reply_sentences,
    )] + _media_parts(media)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        emitted = ""
        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=_gemini_config(max_output_tokens=max_output_tokens),
            ):
                piece = (getattr(chunk, "text", None) or "")
                if not piece:
                    continue
                emitted += piece
                label = f"Gemini · {model}" if attempt == 1 else f"Gemini · {model} · retry recovered"
                yield emitted.strip(), label, False
            if emitted.strip():
                label = f"Gemini · {model}" if attempt == 1 else f"Gemini · {model} · retry recovered"
                yield emitted.strip(), label, True
                return
            last_exc = RuntimeError("empty Gemini stream")
        except Exception as exc:
            last_exc = exc
            if emitted.strip():
                # Preserve usable partial output instead of replacing it with a generic fallback.
                yield emitted.strip(), f"Gemini · {model} · partial", True
                return
            if not _is_transient_error(exc) or attempt >= max_attempts:
                break
            time.sleep(0.12 * attempt)

    reason = type(last_exc).__name__ if last_exc else "Unavailable"
    yield local_reply, f"Local fallback after bounded retry ({reason})", True


def analyze_media_for_jspace(
    media: list[dict] | None,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    domain: str,
    timeout_ms: int = 12000,
) -> list[Concept]:
    """Convert customer-provided image/audio/video into compact JSpace evidence using Gemini."""
    if not media or not api_key:
        return []
    try:
        from google import genai
        from google.genai import types

        client = _cached_client(api_key, timeout_ms)
        prompt = (
            "You are extracting customer-service evidence from media for a research workspace. "
            f"Domain: {domain}. For each attached media item, describe only observable service-relevant evidence "
            "in one concise sentence. Do not infer identity or sensitive traits. Return a JSON array of objects "
            "with keys source, summary, confidence, and optional emotion and emotion_intensity. "
            "source must be image, audio, or video. Only include emotion for audio/video when vocal or visible affect is actually observable. "
            "If emotion is present, choose one of: calm, neutral, curious, hopeful, appreciative, satisfied, relieved, uncertain, confused, anxious, "
            "disappointed, frustrated, angry, impatient, skeptical, distressed, embarrassed. emotion_intensity must be 0 to 1."
        )
        response = client.models.generate_content(
            model=model,
            contents=[prompt] + _media_parts(media),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=600,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        parsed = json.loads(response.text or "[]")
        if isinstance(parsed, dict):
            parsed = parsed.get("items", [])
        concepts: list[Concept] = []
        for i, row in enumerate(parsed if isinstance(parsed, list) else []):
            source = str(row.get("source", "image")).lower()
            if source not in {"image", "audio", "video"}:
                source = "image"
            summary = str(row.get("summary", "")).strip()
            if not summary:
                continue
            try:
                confidence = min(0.99, max(0.4, float(row.get("confidence", 0.82))))
            except Exception:
                confidence = 0.82
            concepts.append(Concept(
                id=f"media_{uuid.uuid4().hex[:10]}",
                name=f"{source}_evidence_{i+1}",
                value=summary,
                sources=[source],
                evidence=[Evidence(source=source, detail=summary)],
                confidence=confidence,
                task_relevance=0.82,
                conflict_importance=0.36,
            ))
            emotion = str(row.get("emotion", "")).lower().strip()
            if source in {"audio", "video"} and emotion in ALLOWED_EMOTIONS:
                try:
                    intensity = min(0.99, max(0.05, float(row.get("emotion_intensity", 0.65))))
                except Exception:
                    intensity = 0.65
                concepts.append(Concept(
                    id=f"media_emotion_{uuid.uuid4().hex[:10]}",
                    name="customer_emotion",
                    value=emotion,
                    sources=[source],
                    evidence=[Evidence(source=source, detail=f"media affect={emotion}; intensity={intensity:.0%}")],
                    confidence=min(0.96, 0.58 + 0.35 * intensity),
                    task_relevance=0.48 + 0.40 * intensity,
                    conflict_importance=0.18 + 0.38 * intensity if emotion in {"frustrated", "angry", "anxious", "impatient", "skeptical", "distressed", "disappointed"} else 0.06,
                ))
                concepts.append(Concept(
                    id=f"media_intensity_{uuid.uuid4().hex[:10]}",
                    name="emotion_intensity",
                    value=f"{intensity:.2f}",
                    sources=[source],
                    evidence=[Evidence(source=source, detail=f"media affect intensity={intensity:.0%}")],
                    confidence=min(0.96, 0.58 + 0.35 * intensity),
                    task_relevance=0.52 + 0.28 * intensity,
                    conflict_importance=0.14 + 0.24 * intensity,
                ))
        return concepts
    except Exception:
        return []


def enhance_scenario_with_gemini(
    scenario: GeneratedScenario,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    channel: str = "text messages",
    timeout_ms: int = 12000,
) -> tuple[GeneratedScenario, str]:
    """Use Gemini to rewrite a controlled scenario into a more varied, realistic case without changing ground truth."""
    if not api_key:
        return scenario, "Curated scenario"
    try:
        from google import genai
        from google.genai import types

        client = _cached_client(api_key, timeout_ms)
        base_turns = [s.customer_turn.text for s in scenario.steps]
        prompt = f"""
Create a realistic customer-service simulation for the following controlled case.
Keep the underlying issue and factual ground truth unchanged, but make the customer's problem and wording feel natural and specific.
The interaction channel is: {channel}.

Domain: {scenario.domain}
Current case title: {scenario.title}
Customer profile: {scenario.customer_profile}
Ground truth that MUST NOT be contradicted: {scenario.hidden_ground_truth}
Number of customer turns required: {len(base_turns)}
Base turn intents, in order: {base_turns}

Requirements:
- Return exactly {len(base_turns)} customer turns in the same logical order.
- Keep each turn conversational and usually 1-3 sentences.
- Do not reveal hidden backend facts before the corresponding customer-facing moment.
- Preserve the customer's emotional trajectory; do not turn every message into anger.
- Vary wording and concrete details to avoid repetitive scenarios.
- Preserve resolution/closing semantics from the base turns. The final customer turn must clearly say they have no other questions or concerns so the agent can close politely.
- Provide a one-sentence problem_summary that a researcher can understand immediately.
""".strip()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ScenarioRewrite,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=560,
            ),
        )
        rewrite = ScenarioRewrite.model_validate_json(response.text)
        if len(rewrite.turns) != len(scenario.steps):
            return scenario, "Curated scenario (Gemini rewrite length mismatch)"
        updated = scenario.model_copy(deep=True)
        updated.title = rewrite.title.strip() or scenario.title
        updated.problem_summary = rewrite.problem_summary.strip()
        # Let Gemini vary the scenario language while keeping the final no-other-concerns
        # turn deterministic so every simulated support conversation closes naturally.
        for step, new_text in zip(updated.steps[:-1], rewrite.turns[:-1]):
            if new_text.strip():
                step.customer_turn.text = new_text.strip()
        updated.generated_by_ai = True
        return updated, f"Gemini · {model}"
    except Exception as exc:
        return scenario, f"Curated fallback ({type(exc).__name__})"


def probe_gemini(*, api_key: str | None, model: str = DEFAULT_MODEL, timeout_ms: int = 12000) -> tuple[bool, str]:
    """Tiny connection check used by Settings; never sends customer data."""
    if not api_key:
        return False, "No GEMINI_API_KEY configured"
    try:
        client = _cached_client(api_key, timeout_ms)
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: OK",
            config=_gemini_config(max_output_tokens=8),
        )
        text = (response.text or "").strip()
        return bool(text), text or "Empty response"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
