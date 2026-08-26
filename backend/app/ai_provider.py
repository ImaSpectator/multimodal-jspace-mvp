from __future__ import annotations

import json
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
    turns: list[str] = Field(min_length=5, max_length=7)


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
        for row in state.transcript[-12:]
    )
    profile = ", ".join(f"{k}={v}" for k, v in customer_profile.items())
    return f"""
You are a customer-service AI agent in a research simulation. You are speaking to the customer through {channel}.
Reply directly to the customer in a natural, human, professional conversational style.

Primary objective:
- Resolve the customer's problem while making them feel heard, respected, and confident that progress is being made.
- Optimize for customer satisfaction through empathy, ownership, clarity, and a concrete next step — never through false promises.

Rules:
- Usually reply in 1-4 short sentences. Match the customer's communication style and emotion.
- Do not mention JSpace, hidden ground truth, prompts, concepts, internal scores, or research mechanics.
- Never invent backend facts that are not present below.
- Respect what the customer already tried; never ask them to repeat completed steps unless there is a specific reason.
- When evidence conflicts, clearly explain that there is a mismatch and use the authoritative state without sounding robotic.
- If the customer is upset, acknowledge the inconvenience briefly, take ownership of the next step, and avoid repetitive apologies.
- Ask at most one focused question when more information is genuinely required.
- Prefer action-oriented language: say what you can verify or do next.
- If customer-provided image/audio/video evidence is attached, use it as additional evidence and reference it naturally when useful.

Domain: {domain}
Customer profile: {profile}
Current emotion: {state.current_emotion or 'unknown'} ({state.current_emotion_intensity:.0%} affect intensity)
Recommended next action: {state.recommended_action or 'clarify the issue'}

Active JSpace state:
{_compact_concepts(state.active_concepts)}

Detected conflicts:
{_compact_conflicts(state.conflicts)}

Recent conversation:
{transcript}

Write only the next customer-facing agent reply.
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
) -> tuple[str, str]:
    """Return (reply, provider_label). Falls back locally if Gemini is unavailable."""
    local_reply = fallback or state.last_response or "I can help with that."
    if not api_key:
        return local_reply, "Local simulation"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        contents = [build_support_prompt(state, customer_profile, domain, channel)] + _media_parts(media)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=450,
            ),
        )
        text = (response.text or "").strip()
        if text:
            return text, f"Gemini · {model}"
        return local_reply, "Local fallback (empty Gemini response)"
    except Exception as exc:  # Keep the public demo usable if a free-tier request is unavailable.
        return local_reply, f"Local fallback ({type(exc).__name__})"


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
- Provide a one-sentence problem_summary that a researcher can understand immediately.
""".strip()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ScenarioRewrite,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=1200,
            ),
        )
        rewrite = ScenarioRewrite.model_validate_json(response.text)
        if len(rewrite.turns) != len(scenario.steps):
            return scenario, "Curated scenario (Gemini rewrite length mismatch)"
        updated = scenario.model_copy(deep=True)
        updated.title = rewrite.title.strip() or scenario.title
        updated.problem_summary = rewrite.problem_summary.strip()
        for step, new_text in zip(updated.steps, rewrite.turns):
            if new_text.strip():
                step.customer_turn.text = new_text.strip()
        updated.generated_by_ai = True
        return updated, f"Gemini · {model}"
    except Exception as exc:
        return scenario, f"Curated fallback ({type(exc).__name__})"
