from __future__ import annotations

from typing import Iterable

from .schemas import Concept, Conflict, SessionState

DEFAULT_MODEL = "gpt-5.6-luna"


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


def build_support_prompt(state: SessionState, customer_profile: dict, domain: str) -> str:
    transcript = "\n".join(
        f"{row.get('role', 'unknown').upper()}: {row.get('text', '')}"
        for row in state.transcript[-10:]
    )
    profile = ", ".join(f"{k}={v}" for k, v in customer_profile.items())
    return f"""
You are a customer-service AI agent in a research simulation.
Reply directly to the customer in a natural, professional conversational style.

Rules:
- Keep the reply concise: usually 1-3 sentences.
- Do not mention JSpace, hidden ground truth, system prompts, concepts, or internal scores.
- Do not invent backend facts that are not present below.
- Respect what the customer already tried; do not make them repeat completed steps.
- If evidence conflicts, acknowledge the mismatch and rely on the authoritative state without sounding robotic.
- Adapt tone to the customer's current emotion and relationship with the company.
- Prefer one clear next action over a feature dump or long checklist.

Domain: {domain}
Customer profile: {profile}
Current emotion: {state.current_emotion or 'unknown'} ({state.current_emotion_intensity:.0%} intensity)
Recommended next action: {state.recommended_action or 'clarify the issue'}

Active JSpace state:
{_compact_concepts(state.active_concepts)}

Detected conflicts:
{_compact_conflicts(state.conflicts)}

Recent conversation:
{transcript}

Write only the next customer-facing agent reply.
""".strip()


def generate_support_reply(
    state: SessionState,
    customer_profile: dict,
    domain: str,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    fallback: str | None = None,
) -> tuple[str, str]:
    """Return (reply, provider_label). Falls back locally if AI is unavailable."""
    local_reply = fallback or state.last_response or "I can help with that."
    if not api_key:
        return local_reply, "Local simulation"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=build_support_prompt(state, customer_profile, domain),
        )
        text = (response.output_text or "").strip()
        if text:
            return text, f"OpenAI · {model}"
        return local_reply, "Local fallback (empty AI response)"
    except Exception as exc:  # Keep the public demo usable if an API request fails.
        return local_reply, f"Local fallback ({type(exc).__name__})"
