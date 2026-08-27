from __future__ import annotations

import base64
import json
import re
import time
import uuid
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Iterable, Iterator

from pydantic import BaseModel, Field

from .schemas import Concept, Conflict, Evidence, GeneratedScenario, SessionState

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
DEFAULT_AUDIO_MODEL = "hy-asr-3.0-preview"
DEFAULT_VIDEO_MODEL = "youtu-vita"
DEFAULT_BASE_URL = "https://tokenhub.tencentmaas.com/v1"


def _language_is_chinese(language: str | None) -> bool:
    value = str(language or "").strip().lower()
    return value.startswith(("zh", "chinese", "simplified chinese", "简体中文", "中文"))
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


def _last_agent_reply(state: SessionState) -> str:
    return next(
        (str(row.get("text", "")).strip() for row in reversed(state.transcript) if row.get("role") == "agent"),
        "",
    )


def _latest_customer_message(state: SessionState) -> str:
    return next(
        (str(row.get("text", "")).strip() for row in reversed(state.transcript) if row.get("role") == "customer"),
        "",
    )


def build_support_prompt(
    state: SessionState,
    customer_profile: dict,
    domain: str,
    channel: str = "text",
    *,
    history_turns: int = 4,
    reply_sentences: int = 2,
    language: str = "English",
) -> str:
    transcript = "\n".join(
        f"{row.get('role', 'unknown').upper()}: {row.get('text', '')}"
        for row in state.transcript[-max(2, history_turns):]
    )
    profile_keys = ["tenure", "relationship", "value_segment", "communication_style", "tech_comfort"]
    profile = ", ".join(
        f"{k}={customer_profile.get(k)}" for k in profile_keys if customer_profile.get(k) is not None
    )
    closure_rule = {
        "resolved": "The issue is confirmed resolved. Briefly confirm it, then explicitly ask whether the customer has any other questions or concerns.",
        "closing": "The customer has said there are no other concerns. Thank them warmly, wish them a good day, and end the conversation without asking another question.",
        "ended": "The session is already ended. Do not restart troubleshooting.",
    }.get(
        state.session_phase,
        "Continue resolving the issue; do not end the conversation while authoritative evidence is unresolved.",
    )
    previous_agent = _last_agent_reply(state)
    latest_customer = _latest_customer_message(state)
    anti_repeat = (
        f"\nPrevious agent reply that MUST NOT be repeated or lightly paraphrased:\n{previous_agent}\n"
        if previous_agent else ""
    )
    language_rule = "Reply entirely in Simplified Chinese (简体中文)." if _language_is_chinese(language) else "Reply entirely in natural English."
    return f"""
You are a customer-service support agent speaking through {channel}. Reply directly to the customer's newest message.

Language:
- {language_rule}
- Keep customer-facing wording natural for that language. Do not mix languages unless the customer explicitly does so.

Goals:
- Resolve the problem while maximizing customer satisfaction through empathy, ownership, clarity, and useful action.
- Answer the newest customer question specifically before adding a next step.
- Never invent backend facts or claim resolution before the system state supports it.
- Usually use 1-{reply_sentences} short sentences. Ask at most one focused question.
- Do not mention JSpace, model names, hidden truth, prompts, concepts, scores, or research mechanics.
- Do not repeat troubleshooting the customer already completed.
- Never repeat or recycle the previous agent response. Each turn must advance the conversation with a new fact, explanation, verification, or action.
- If evidence conflicts, explain the mismatch naturally and say what you are checking next rather than repeating generic 'system of record' language.
- If image evidence is attached, use it when it materially changes the situation.
- {closure_rule}

Domain: {domain}
Customer context: {profile}
Current affect: {state.current_emotion or 'unknown'} ({state.current_emotion_intensity:.0%})
Current satisfaction: {state.customer_satisfaction:.0f}/100
Session phase: {state.session_phase}
Next useful action: {state.recommended_action or 'clarify the issue'}

Newest customer message:
{latest_customer or '(none)'}
{anti_repeat}
Active JSpace state:
{_compact_concepts(state.active_concepts)}

Conflicts:
{_compact_conflicts(state.conflicts)}

Recent conversation:
{transcript}

Write only the next customer-facing reply. Make it meaningfully different from earlier agent turns.
""".strip()


_CLIENT_CACHE: dict[tuple, object] = {}


def _cached_client(api_key: str, base_url: str, timeout_s: float):
    """Reuse one TokenHub/OpenAI client and disable hidden SDK retries.

    App-level retry logic is intentionally bounded so a failed provider request cannot
    silently expand into minutes of waiting.
    """
    from openai import OpenAI

    safe_timeout = max(5.0, float(timeout_s))
    key = (api_key, base_url.rstrip("/"), safe_timeout, id(OpenAI))
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=safe_timeout,
            max_retries=0,
        )
    return _CLIENT_CACHE[key]


def _clear_cached_clients() -> None:
    _CLIENT_CACHE.clear()


_cached_client.cache_clear = _clear_cached_clients  # type: ignore[attr-defined]


def _is_transient_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in [
        "503", "502", "504", "500", "429", "rate limit", "unavailable", "temporarily",
        "timeout", "timed out", "connection", "internal", "server error", "service unavailable",
    ])


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)).strip()


def _too_similar_to_previous(state: SessionState, candidate: str, *, threshold: float = 0.86) -> bool:
    previous = _normalize_text(_last_agent_reply(state))
    current = _normalize_text(candidate)
    if not previous or not current:
        return False
    if previous == current:
        return True
    # Exact-ish opening repeats are particularly noticeable in a live chat.
    if len(previous) >= 45 and len(current) >= 45 and previous[:45] == current[:45]:
        return True
    return SequenceMatcher(None, previous, current).ratio() >= threshold


def _fallback_reply(state: SessionState, preferred: str | None = None, *, language: str = "English") -> str:
    """Contextual backup responder with enough variants to avoid visible loops."""
    previous_agents = [
        str(row.get("text", "")).strip()
        for row in state.transcript
        if row.get("role") == "agent" and str(row.get("text", "")).strip()
    ]
    preferred = (preferred or "").strip()
    if preferred and preferred not in previous_agents:
        return preferred

    chinese = _language_is_chinese(language)
    if chinese:
        variants = {
            "resolve_conflict": [
                "你看到的状态和公司系统记录仍然不一致。我正在核对真正决定结果的状态字段，这样可以明确告诉你具体还卡在哪里。",
                "我还不能确认问题已经解决，因为两边记录仍有冲突。下一步我会核对权威状态和对应的阻塞原因。",
                "目前前台显示与后台记录不一致。我会直接定位阻止流程完成的系统状态，不会让你重复已经做过的步骤。",
            ],
            "act_on_root_cause": [
                "我已经定位到一个具体的底层阻塞原因。接下来会针对这个原因处理，而不是继续让你做通用排查。",
                "现在已经有明确的根因方向，下一步应该是针对性修复，而不是让你再次重试。",
                "我已经把问题缩小到一个具体阻塞点，接下来会根据这个状态采取修复动作。",
            ],
            "avoid_repetition": [
                "你已经完成的步骤我都有记录，我不会让你再重复。接下来我会检查一个新的系统侧信号。",
                "前面的排查结果我会保留在上下文里，下一步会做新的检查，而不是再次让你重试同样的步骤。",
            ],
            "confirm_resolution": [
                "系统现在已经显示问题解决。结束前还有其他问题或需要我继续确认的地方吗？",
                "我现在可以确认系统状态已经恢复正常。还有其他问题需要我一起处理吗？",
            ],
            "close_session": [
                "很高兴这次问题已经处理好。感谢你联系支持，祝你今天顺利。",
                "这边已经全部处理完成。感谢你的耐心，祝你有愉快的一天。",
            ],
            "default": [
                "我会继续从当前状态往下排查，并给你一个具体的新信息或下一步，而不是重复之前的回答。",
                "我已经保留了你刚才提供的信息，接下来会核对最能推进问题解决的系统状态。",
            ],
        }
    else:
        variants = {
        "resolve_conflict": [
            "What you’re seeing and the company record still disagree. I’m checking the specific status field that controls the outcome so I can tell you exactly what has to change.",
            "I haven’t confirmed this as resolved yet because the records still conflict. My next check is the authoritative status and the blocker attached to it.",
            "There’s still a mismatch between your side and the backend. I’m narrowing it to the exact system state that is preventing completion rather than asking you to repeat anything.",
            "The evidence is still inconsistent, so I don’t want to guess. I’m verifying which record is current and what action would actually clear the blocker.",
        ],
        "act_on_root_cause": [
            "I’ve narrowed this to a concrete underlying blocker. I’m working from that cause now and will explain the fix rather than sending you through generic troubleshooting.",
            "We now have a likely root cause, so the next step is targeted remediation rather than another retry on your side.",
            "The issue is no longer just a generic failure—I have a specific blocker to work from. I’m using that to determine the corrective action now.",
            "I’ve isolated the likely cause. I’m moving to the fix for that condition instead of repeating steps that don’t address it.",
        ],
        "avoid_repetition": [
            "I have the steps you already completed, so I won’t make you repeat them. I’m moving to the next diagnostic check that can actually change the outcome.",
            "Your previous troubleshooting is noted. I’m skipping those repeated steps and checking the next system-side cause instead.",
            "You’ve already done the basic checks, so I’m not sending you through them again. I’m escalating the investigation to the next useful signal.",
            "I’m keeping the work you already did in context. The next step will be a new check, not another retry of the same thing.",
        ],
        "confirm_resolution": [
            "The system now reflects the resolution. Is there anything else you’d like me to check before we wrap up?",
            "I can now confirm the issue is resolved in the system. Do you have any other questions or concerns I can help with?",
            "The updated status is now consistent and resolved. Before we close, is there anything else you want me to look into?",
        ],
        "close_session": [
            "I’m glad we could get this sorted out. Thanks for contacting support, and I hope you have a great day.",
            "Everything is wrapped up on my side. Thanks for reaching out, and have a great rest of your day.",
            "That completes the case. Thanks for your time today, and I hope the rest of your day goes smoothly.",
        ],
        "investigate": [
            "I’m checking the current account and system details now so I can give you a concrete next step rather than a generic answer.",
            "This still needs investigation, so I’m keeping it open and checking the specific blocker before I call it resolved.",
            "I’m reviewing the latest system state and the most relevant evidence so the next response gives you something actionable.",
            "I’m tracing the issue to the next verifiable point in the process so I can tell you what is actually preventing completion.",
        ],
    }
    options = variants.get(state.recommended_action_code or "", variants.get("default", [
        "I’m reviewing the current details so I can give you a new, useful next step without making you repeat yourself.",
        "I’m checking the latest state now and will focus the next step on what can actually move the issue forward.",
        "I’m comparing the latest evidence so I can answer your question directly and avoid recycling the same troubleshooting advice.",
    ]))
    # Choose the first variant not already used in this transcript.
    for option in options:
        if option not in previous_agents:
            return option
    return options[len(previous_agents) % len(options)]


def _image_content_parts(media: list[dict] | None, prompt: str) -> list[dict] | str:
    images = []
    for item in media or []:
        mime = str(item.get("mime_type") or "").lower()
        data = item.get("data")
        if not data or not mime.startswith("image/"):
            continue
        b64 = base64.b64encode(data).decode("ascii")
        images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}] + images[:4]


def _chat_messages(state: SessionState, prompt: str, media: list[dict] | None = None) -> list[dict]:
    system = (
        "You are a concise, high-quality customer-service agent. Follow the supplied case state exactly, "
        "answer the newest customer question directly, and never repeat the previous agent response."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _image_content_parts(media, prompt)},
    ]


def _request_kwargs(*, model: str, messages: list[dict], max_tokens: int, stream: bool) -> dict:
    # TokenHub DeepSeek defaults to thinking enabled. Customer-service chat benefits from
    # disabling it for lower latency; this follows TokenHub's OpenAI-compatible DeepSeek guide.
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def generate_support_reply(
    state: SessionState,
    customer_profile: dict,
    domain: str,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    fallback: str | None = None,
    channel: str = "text",
    media: list[dict] | None = None,
    max_attempts: int = 2,
    timeout_s: float = 12.0,
    history_turns: int = 4,
    max_output_tokens: int = 140,
    reply_sentences: int = 2,
    language: str = "English",
) -> tuple[str, str]:
    """Return (reply, provider label) through Tencent TokenHub DeepSeek."""
    local_reply = _fallback_reply(state, fallback, language=language)
    if not api_key:
        return local_reply, "Local simulation"

    client = _cached_client(api_key, base_url, timeout_s)
    base_prompt = build_support_prompt(
        state, customer_profile, domain, channel,
        history_turns=history_turns, reply_sentences=reply_sentences, language=language,
    )
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += "\n\nRETRY INSTRUCTION: The earlier attempt failed or was repetitive. Give a fresh, specific response to the newest customer question."
        try:
            response = client.chat.completions.create(**_request_kwargs(
                model=model,
                messages=_chat_messages(state, prompt, media),
                max_tokens=max_output_tokens,
                stream=False,
            ))
            text = (response.choices[0].message.content or "").strip()
            if text and not _too_similar_to_previous(state, text):
                label = f"DeepSeek · {model}" if attempt == 1 else f"DeepSeek · {model} · retry recovered"
                return text, label
            last_exc = RuntimeError("empty or repetitive DeepSeek response")
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
    base_url: str = DEFAULT_BASE_URL,
    fallback: str | None = None,
    channel: str = "text",
    media: list[dict] | None = None,
    max_attempts: int = 2,
    timeout_s: float = 12.0,
    history_turns: int = 4,
    max_output_tokens: int = 140,
    reply_sentences: int = 2,
    language: str = "English",
) -> Iterator[tuple[str, str, bool]]:
    """Yield progressively growing DeepSeek output with bounded retry and repeat protection."""
    local_reply = _fallback_reply(state, fallback, language=language)
    if not api_key:
        yield local_reply, "Local simulation", True
        return

    client = _cached_client(api_key, base_url, timeout_s)
    base_prompt = build_support_prompt(
        state, customer_profile, domain, channel,
        history_turns=history_turns, reply_sentences=reply_sentences, language=language,
    )
    last_exc: Exception | None = None
    previous = _normalize_text(_last_agent_reply(state))

    for attempt in range(1, max(1, max_attempts) + 1):
        emitted = ""
        buffered = ""
        started_ui = False
        prompt = base_prompt
        if attempt > 1:
            prompt += "\n\nRETRY INSTRUCTION: Give a new response that directly answers the latest customer message. Do not reuse the prior agent wording."
        try:
            stream = client.chat.completions.create(**_request_kwargs(
                model=model,
                messages=_chat_messages(state, prompt, media),
                max_tokens=max_output_tokens,
                stream=True,
            ))
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                piece = (getattr(delta, "content", None) or "") if delta is not None else ""
                if not piece:
                    continue
                buffered += piece
                # Hold only a short prefix so we can catch the exact repeated-fallback-looking
                # opening before it flashes on screen. Normal replies start streaming quickly.
                if not started_ui and len(buffered) < 56:
                    continue
                if not started_ui:
                    norm = _normalize_text(buffered)
                    if previous and len(norm) >= 30 and SequenceMatcher(None, previous[:len(norm)], norm).ratio() >= 0.90:
                        last_exc = RuntimeError("repetitive DeepSeek stream")
                        buffered = ""
                        emitted = ""
                        break
                    emitted = buffered
                    started_ui = True
                else:
                    emitted += piece
                label = f"DeepSeek · {model}" if attempt == 1 else f"DeepSeek · {model} · retry recovered"
                yield emitted.strip(), label, False
            else:
                # Natural stream completion.
                if not started_ui and buffered.strip():
                    emitted = buffered
                if emitted.strip() and not _too_similar_to_previous(state, emitted):
                    label = f"DeepSeek · {model}" if attempt == 1 else f"DeepSeek · {model} · retry recovered"
                    yield emitted.strip(), label, True
                    return
                if not last_exc:
                    last_exc = RuntimeError("empty or repetitive DeepSeek stream")

            # Break in the loop above means a repetitive prefix was detected before display.
            if emitted.strip() and started_ui:
                yield emitted.strip(), f"DeepSeek · {model} · partial", True
                return
        except Exception as exc:
            last_exc = exc
            if emitted.strip():
                yield emitted.strip(), f"DeepSeek · {model} · partial", True
                return
            if not _is_transient_error(exc) or attempt >= max_attempts:
                break

        if attempt < max_attempts:
            time.sleep(0.12 * attempt)

    reason = type(last_exc).__name__ if last_exc else "Unavailable"
    yield local_reply, f"Local fallback after bounded retry ({reason})", True


def _media_error_concept(source: str, detail: str) -> Concept:
    return Concept(
        id=f"media_{source}_error_{uuid.uuid4().hex[:10]}",
        name=f"{source}_analysis_status",
        value="analysis unavailable",
        sources=[source],
        evidence=[Evidence(source=source, detail=detail)],
        confidence=0.99,
        task_relevance=0.42,
        conflict_importance=0.06,
    )


def _audio_format(item: dict) -> str:
    mime = str(item.get("mime_type") or "").lower()
    name = str(item.get("name") or "").lower()
    known = {
        "audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3",
        "audio/mp3": "mp3", "audio/ogg": "ogg", "audio/opus": "ogg",
        "audio/mp4": "auto", "audio/x-m4a": "auto",
    }
    if mime in known:
        return known[mime]
    for ext in ("wav", "mp3", "ogg", "m4a", "pcm"):
        if name.endswith("." + ext):
            return ext
    return "auto"


def transcribe_audio_with_hyasr(
    item: dict,
    *,
    api_key: str | None,
    model: str = DEFAULT_AUDIO_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 25.0,
) -> tuple[str | None, str]:
    """Transcribe one uploaded audio file through TokenHub Hy-ASR sync API.

    TokenHub accepts standard Base64 audio directly, so Streamlit uploads do not need
    to be published to a separate object store first.
    """
    if not api_key:
        return None, "No TOKENHUB_API_KEY configured"
    data = item.get("data")
    if not data:
        return None, "Audio file is empty"
    payload = {
        "model": model,
        "data": base64.b64encode(data).decode("ascii"),
        "voice_encode_format": _audio_format(item),
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/wand/asrproxy/sync_transcribe",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(8.0, float(timeout_s))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        output = parsed.get("output") or {}
        text = str(output.get("text") or "").strip()
        if text:
            lang = str(output.get("source") or "auto")
            duration = output.get("duration_ms")
            suffix = f" · {duration} ms" if duration is not None else ""
            return text, f"Hy-ASR · {model} · language={lang}{suffix}"
        return None, f"Hy-ASR · {model} returned no transcript"
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        return None, f"Hy-ASR HTTP {exc.code}: {body or exc.reason}"
    except Exception as exc:
        return None, f"Hy-ASR {type(exc).__name__}: {exc}"



def analyze_video_with_youtuvita(
    item: dict,
    *,
    api_key: str | None,
    model: str = DEFAULT_VIDEO_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    domain: str,
    timeout_s: float = 30.0,
) -> tuple[list[Concept], str]:
    """Analyze one uploaded video with YT-VITA.

    Official YT-VITA docs specify video_url input. For local Streamlit uploads this
    build uses an OpenAI-style data URL as a best-effort bridge. If TokenHub rejects
    data URLs for YT-VITA, the caller gets an explicit unavailable concept instead of
    fabricated observations. Public URL input can be added later without changing the
    downstream JSpace contract.
    """
    if not api_key:
        return [], "No TOKENHUB_API_KEY configured"
    direct_url = str(item.get("url") or "").strip()
    data = item.get("data")
    if not direct_url and not data:
        return [], "Video file is empty"
    mime = str(item.get("mime_type") or "video/mp4").lower()
    if data and len(data) > 100 * 1024 * 1024:
        return [], "YT-VITA upload exceeds the documented 100 MB video limit"
    if direct_url:
        video_url = direct_url
    else:
        b64 = base64.b64encode(data).decode("ascii")
        video_url = f"data:{mime};base64,{b64}"
    prompt = (
        "Analyze this customer-service video, including both the visible scene and the spoken/audio content. "
        f"Domain: {domain}. Return ONLY a compact JSON object with keys summary, visible_evidence, spoken_content. "
        "summary must be one service-relevant sentence. visible_evidence and spoken_content may be empty strings. "
        "Do not infer identity, protected traits, health status, or facts not supported by the video."
    )
    try:
        client = _cached_client(api_key, base_url, timeout_s)
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=420,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {"summary": raw}
        summary = str(parsed.get("summary") or "").strip()
        visible = str(parsed.get("visible_evidence") or "").strip()
        spoken = str(parsed.get("spoken_content") or "").strip()
        concepts: list[Concept] = []
        if summary:
            concepts.append(Concept(
                id=f"media_video_{uuid.uuid4().hex[:10]}",
                name="video_summary",
                value=summary,
                sources=["video"],
                evidence=[Evidence(source="video", detail=f"YT-VITA summary: {summary}")],
                confidence=0.86,
                task_relevance=0.88,
                conflict_importance=0.42,
            ))
        if visible:
            concepts.append(Concept(
                id=f"media_video_visual_{uuid.uuid4().hex[:10]}",
                name="video_visible_evidence",
                value=visible,
                sources=["video"],
                evidence=[Evidence(source="video", detail=f"YT-VITA visible evidence: {visible}")],
                confidence=0.84,
                task_relevance=0.83,
                conflict_importance=0.46,
            ))
        if spoken:
            concepts.append(Concept(
                id=f"media_video_spoken_{uuid.uuid4().hex[:10]}",
                name="video_spoken_content",
                value=spoken,
                sources=["video"],
                evidence=[Evidence(source="video", detail=f"YT-VITA audio-track understanding: {spoken}")],
                confidence=0.80,
                task_relevance=0.84,
                conflict_importance=0.34,
            ))
        return concepts, f"YT-VITA · {model}"
    except Exception as exc:
        return [], f"YT-VITA {type(exc).__name__}: {exc}"


def analyze_media_for_jspace(
    media: list[dict] | None,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    audio_model: str = DEFAULT_AUDIO_MODEL,
    video_model: str = DEFAULT_VIDEO_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    domain: str,
    timeout_s: float = 12.0,
) -> list[Concept]:
    """Route media by modality: DeepSeek=image, Hy-ASR=audio, YT-VITA=video."""
    if not media:
        return []
    concepts: list[Concept] = []
    images = [m for m in media if str(m.get("mime_type") or "").lower().startswith("image/") and m.get("data")]
    audios = [m for m in media if str(m.get("mime_type") or "").lower().startswith("audio/") and m.get("data")]
    videos = [m for m in media if str(m.get("mime_type") or "").lower().startswith("video/") and (m.get("data") or m.get("url"))]

    if images and api_key:
        prompt = (
            "Extract customer-service evidence from the attached image(s). "
            f"Domain: {domain}. Return ONLY JSON as an object with an 'items' array. Each item must contain "
            "summary (one observable service-relevant sentence) and confidence (0 to 1). Do not infer identity, "
            "health status, protected traits, or anything not directly visible."
        )
        try:
            client = _cached_client(api_key, base_url, timeout_s)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _image_content_parts(images, prompt)}],
                max_tokens=500,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (response.choices[0].message.content or "{}").strip()
            parsed = json.loads(raw)
            rows = parsed.get("items", []) if isinstance(parsed, dict) else []
            for i, row in enumerate(rows if isinstance(rows, list) else []):
                summary = str(row.get("summary", "")).strip()
                if not summary:
                    continue
                try:
                    confidence = min(0.99, max(0.4, float(row.get("confidence", 0.82))))
                except Exception:
                    confidence = 0.82
                concepts.append(Concept(
                    id=f"media_image_{uuid.uuid4().hex[:10]}",
                    name=f"image_evidence_{i+1}",
                    value=summary,
                    sources=["image"],
                    evidence=[Evidence(source="image", detail=f"DeepSeek Vision: {summary}")],
                    confidence=confidence,
                    task_relevance=0.84,
                    conflict_importance=0.38,
                ))
        except Exception as exc:
            concepts.append(_media_error_concept("image", f"DeepSeek Vision analysis failed: {type(exc).__name__}: {exc}"))
    elif images:
        concepts.append(_media_error_concept("image", "Image attached but TOKENHUB_API_KEY is not configured."))

    for item in audios:
        transcript, provider = transcribe_audio_with_hyasr(
            item, api_key=api_key, model=audio_model, base_url=base_url, timeout_s=max(timeout_s, 20.0),
        )
        if transcript:
            concepts.append(Concept(
                id=f"media_audio_{uuid.uuid4().hex[:10]}",
                name="audio_transcript",
                value=transcript,
                sources=["audio"],
                evidence=[Evidence(source="audio", detail=f"{provider}: {transcript}")],
                confidence=0.92,
                task_relevance=0.94,
                conflict_importance=0.38,
            ))
        else:
            concepts.append(_media_error_concept("audio", provider))

    # YT-VITA supports one video per request; analyze uploads independently so the UI remains predictable.
    for item in videos:
        video_concepts, provider = analyze_video_with_youtuvita(
            item, api_key=api_key, model=video_model, base_url=base_url, domain=domain,
            timeout_s=max(timeout_s, 30.0),
        )
        if video_concepts:
            concepts.extend(video_concepts)
        else:
            concepts.append(_media_error_concept("video", provider))

    return concepts

def enhance_scenario_with_deepseek(
    scenario: GeneratedScenario,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    channel: str = "text messages",
    timeout_s: float = 12.0,
    language: str = "English",
) -> tuple[GeneratedScenario, str]:
    """Rewrite controlled scenario language with DeepSeek without changing ground truth."""
    if not api_key:
        return scenario, "Curated scenario"
    try:
        client = _cached_client(api_key, base_url, timeout_s)
        base_turns = [s.customer_turn.text for s in scenario.steps]
        language_rule = "Write title, problem_summary, and every turn entirely in Simplified Chinese (简体中文)." if _language_is_chinese(language) else "Write title, problem_summary, and every turn entirely in natural English."
        prompt = f"""
Create a realistic customer-service simulation for this controlled case. Keep factual ground truth unchanged.
Language requirement: {language_rule}
Interaction channel: {channel}
Domain: {scenario.domain}
Current title: {scenario.title}
Customer profile: {scenario.customer_profile}
Ground truth that MUST NOT be contradicted: {scenario.hidden_ground_truth}
Number of customer turns required: {len(base_turns)}
Base turn intents in order: {base_turns}

Return ONLY JSON with keys title, problem_summary, turns.
- turns must contain exactly {len(base_turns)} strings in the same logical order.
- Keep each turn conversational and 1-3 sentences.
- Preserve the emotional trajectory; do not make every message angry.
- Vary wording/details to avoid repetitive scenarios.
- Preserve resolution and closing semantics.
- The final customer turn must clearly say there are no other questions or concerns.
""".strip()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=620,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        parsed = json.loads((response.choices[0].message.content or "{}").strip())
        rewrite = ScenarioRewrite.model_validate(parsed)
        if len(rewrite.turns) != len(scenario.steps):
            return scenario, "Curated scenario (DeepSeek rewrite length mismatch)"
        updated = scenario.model_copy(deep=True)
        updated.title = rewrite.title.strip() or scenario.title
        updated.problem_summary = rewrite.problem_summary.strip() or scenario.problem_summary
        for step, new_text in zip(updated.steps, rewrite.turns):
            if new_text.strip():
                step.customer_turn.text = new_text.strip()
        updated.generated_by_ai = True
        return updated, f"DeepSeek · {model}"
    except Exception as exc:
        return scenario, f"Curated fallback ({type(exc).__name__})"


def probe_deepseek(
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 12.0,
) -> tuple[bool, str]:
    """Tiny TokenHub connection check; never sends customer data."""
    if not api_key:
        return False, "No TOKENHUB_API_KEY configured"
    try:
        client = _cached_client(api_key, base_url, timeout_s)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=8,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (response.choices[0].message.content or "").strip()
        return bool(text), text or "Empty response"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
