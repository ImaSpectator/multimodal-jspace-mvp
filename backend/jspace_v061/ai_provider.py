from __future__ import annotations

import json
import time
import uuid
from typing import Iterable

from pydantic import BaseModel, Field

from .schemas import Concept, Conflict, Evidence, GeneratedScenario, SessionState

DEFAULT_MODEL = "gemini-3.7-flash"
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


def build_support_prompt(state: SessionState, customer_profile: dict, domain: str, channel: str = "text") -> str:
    transcript = "\n".join(
        f"{row.get('role', 'unknown').upper()}: {row.get('text', '')}"
        for row in state.transcript[-6:]
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
- Usually use 1-3 short sentences. Ask at most one focused question.
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
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Return (reply, provider label), retrying transient Gemini failures first."""
    local_reply = fallback or state.last_response or "I can help with that."
    if not api_key:
        return local_reply, "Local simulation"

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [build_support_prompt(state, customer_profile, domain, channel)] + _media_parts(media)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    max_output_tokens=260,
                    temperature=0.45,
                ),
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
            time.sleep(0.25 * attempt)

    reason = type(last_exc).__name__ if last_exc else "Unavailable"
    return local_reply, f"Local fallback after retry ({reason})"


def analyze_media_for_jspace(
    media: list[dict] | None,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    domain: str,
) -> list[Concept]:
    """Convert customer-provided image/audio/video into compact JSpace evidence using Gemini."""
    if not media or not api_key:
        return []
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
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
) -> tuple[GeneratedScenario, str]:
    """Use Gemini to rewrite a controlled scenario into a more varied, realistic case without changing ground truth."""
    if not api_key:
        return scenario, "Curated scenario"
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
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
                max_output_tokens=900,
                temperature=0.75,
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
