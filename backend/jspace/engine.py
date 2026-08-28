from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Iterable

from .schemas import BackendEvent, Concept, Conflict, CustomerTurn, Evidence, ImageObservation, SessionState


NEGATIVE_EMOTIONS = {"uncertain", "confused", "anxious", "disappointed", "frustrated", "angry", "impatient", "skeptical", "distressed"}
RESOLVED_VISUAL_VALUES = {
    "appears successful", "delivered", "wifi visible", "partial access", "cancellation requested",
    "new itinerary visible", "return completed", "100% progress", "connected", "workspace visible",
    "adjusted badge", "confirmed", "active", "upgrade visible", "cancelled", "ticket visible",
    "upgrade complete", "completed",
}

CONFLICT_VALUE_ZH = {
    "appears successful": "看起来已成功",
    "delivered": "已送达",
    "wifi visible": "Wi-Fi 可见",
    "partial access": "部分访问",
    "cancellation requested": "已请求取消",
    "new itinerary visible": "新行程已显示",
    "return completed": "退货已完成",
    "100% progress": "进度显示 100%",
    "connected": "已连接",
    "workspace visible": "工作区可见",
    "adjusted badge": "已显示调整标记",
    "confirmed": "已确认",
    "active": "有效",
    "upgrade visible": "升级信息可见",
    "cancelled": "已取消",
    "ticket visible": "票券可见",
    "upgrade complete": "升级已完成",
    "completed": "已完成",
}

CONFLICT_EMOTION_ZH = {
    "uncertain": "不确定", "confused": "困惑", "anxious": "焦虑", "disappointed": "失望",
    "frustrated": "沮丧", "angry": "生气", "impatient": "不耐烦", "skeptical": "怀疑",
    "distressed": "难受",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _concept(name: str, value: str, source: str, evidence: str, *, confidence: float, relevance: float,
             conflict_importance: float = 0.0, status: str = "supported") -> Concept:
    return Concept(
        id=_cid(name.replace(" ", "_")),
        name=name,
        value=value,
        sources=[source],
        evidence=[Evidence(source=source, detail=evidence)],
        confidence=max(0.0, min(1.0, confidence)),
        task_relevance=max(0.0, min(1.0, relevance)),
        conflict_importance=max(0.0, min(1.0, conflict_importance)),
        status=status,
    )


def infer_text_emotion(text: str) -> tuple[str, float]:
    low = text.lower()
    exclamations = text.count("!")
    question_marks = text.count("?")
    patterns = [
        ("angry", ["ridiculous", "unacceptable", "furious", "angry", "this is insane", "terrible"]),
        ("distressed", ["desperate", "emergency", "please help", "can't deal", "really scared"]),
        ("frustrated", ["frustrated", "again", "already", "still not", "keeps", "why is this", "fed up"]),
        ("impatient", ["hurry", "right now", "quickly", "how long", "immediately", "today"]),
        ("skeptical", ["don't believe", "are you sure", "supposedly", "can i trust", "actually correct"]),
        ("anxious", ["worried", "anxious", "nervous", "concerned", "deadline", "tomorrow"]),
        ("confused", ["confused", "don't understand", "doesn't make sense", "what does", "why does"]),
        ("disappointed", ["disappointed", "expected better", "let down"]),
        ("relieved", ["relieved", "thank goodness", "finally works", "that's a relief"]),
        ("appreciative", ["thank you", "thanks", "appreciate"]),
        ("hopeful", ["hopefully", "hope this", "sounds promising"]),
        ("satisfied", ["perfect", "great", "that works", "all set"]),
        ("embarrassed", ["my mistake", "sorry, i", "embarrassing"]),
    ]
    for emotion, terms in patterns:
        if any(term in low for term in terms):
            base = 0.58
            if emotion in {"angry", "distressed", "frustrated"}:
                base = 0.68
            intensity = min(0.98, base + 0.05 * exclamations + 0.02 * question_marks + min(len(text) / 800, 0.12))
            return emotion, round(intensity, 2)
    if question_marks >= 2:
        return "uncertain", min(0.82, round(0.48 + 0.05 * question_marks, 2))
    return "neutral", round(min(0.65, 0.34 + len(text) / 600), 2)


def extract_from_turn(turn: CustomerTurn) -> list[Concept]:
    text = turn.text.strip()
    low = text.lower()
    out: list[Concept] = []

    resolved_phrases = [
        "it worked", "it's working", "its working", "fixed now", "resolved", "all good", "fine now",
        "looks fixed", "looks good", "got back in", "cancelled now", "canceled now", "completed now",
        "delivered now", "went through", "successful now",
    ]
    unresolved_phrases = [
        "still not", "still failing", "not fixed", "still broken", "doesn't work", "doesnt work", "still locked",
        "still missing", "never arrived", "still pending", "not here", "keeps dropping", "can't", "cannot",
    ]
    if any(k in low for k in resolved_phrases):
        out.append(_concept("customer_belief_status", "resolved", "text", text,
                            confidence=0.84, relevance=0.92, conflict_importance=0.55))
    elif any(k in low for k in unresolved_phrases):
        out.append(_concept("customer_belief_status", "unresolved", "text", text,
                            confidence=0.90, relevance=0.94, conflict_importance=0.25))

    m = re.search(r"(?:tried|attempted|called|restarted|reset|contacted)(?: it)?\s+(\w+|\d+)\s+times", low)
    if m:
        raw = m.group(1)
        word_map = {"once": 1, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        n = word_map.get(raw, int(raw) if raw.isdigit() else None)
        if n:
            out.append(_concept("retry_count", str(n), "text", text, confidence=0.94, relevance=0.90))
            out.append(_concept("avoid_repeat_action", "avoid repeating completed troubleshooting", "derived", text,
                                confidence=0.90, relevance=0.96))
    if any(k in low for k in ["already tried", "already did", "already restarted", "already reset", "already contacted", "asked me that already"]):
        out.append(_concept("prior_effort", "customer has already completed troubleshooting", "text", text,
                            confidence=0.90, relevance=0.92))
        out.append(_concept("avoid_repeat_action", "avoid repeating completed troubleshooting", "derived", text,
                            confidence=0.91, relevance=0.97))

    domain_patterns = {
        "payment": ["payment", "pay", "card", "transaction", "checkout", "declined"],
        "delivery": ["package", "delivery", "shipment", "tracking", "courier", "arrived"],
        "internet": ["internet", "wifi", "wi-fi", "router", "modem", "connection", "outage"],
        "account_access": ["login", "log in", "password", "locked", "account access", "verification"],
        "subscription": ["subscription", "cancel", "membership", "renewal", "charged again"],
        "travel": ["flight", "booking", "itinerary", "ticketed", "reservation", "seat"],
        "return_refund": ["refund", "return", "returned", "money back"],
        "insurance_claim": ["claim", "insurance", "adjuster", "proof-of-loss"],
        "device_support": ["device", "speaker", "firmware", "smart", "disconnecting"],
        "software_saas": ["workspace", "dashboard", "permission", "access denied", "admin"],
        "utilities": ["electricity", "utility", "meter", "bill", "reading"],
        "healthcare_appointment": ["appointment", "patient portal", "clinician", "visit"],
        "banking_fraud": ["fraud", "dispute", "don't recognize", "card active", "unauthorized"],
        "hotel_hospitality": ["hotel", "room", "check in", "upgrade"],
        "rideshare": ["driver", "ride", "pickup", "cancelled ride"],
        "event_ticketing": ["concert", "event", "barcode", "ticket transfer"],
        "telecom_mobile": ["mobile plan", "data limit", "carrier", "network settings"],
        "marketplace_dispute": ["marketplace", "seller", "replacement", "case", "dispute"],
    }
    for domain, patterns in domain_patterns.items():
        if any(p in low for p in patterns):
            out.append(_concept("customer_domain", domain, "text", text, confidence=0.90, relevance=0.99))
            break

    emotion = turn.emotion
    intensity = turn.emotion_intensity
    emotion_confidence = 0.56 + 0.39 * intensity
    emotion_relevance = 0.45 + 0.45 * intensity
    conflict_importance = (0.15 + 0.45 * intensity) if emotion in NEGATIVE_EMOTIONS else 0.05
    evidence = f"emotion={emotion}; intensity={intensity:.0%}"
    if turn.nonverbal_cue:
        evidence += f"; cue={turn.nonverbal_cue}"
    affect_source = turn.affect_source
    out.append(_concept("customer_emotion", emotion, affect_source, evidence,
                        confidence=emotion_confidence, relevance=emotion_relevance,
                        conflict_importance=conflict_importance))
    out.append(_concept("emotion_intensity", f"{intensity:.2f}", affect_source, evidence,
                        confidence=emotion_confidence, relevance=0.58 + 0.25 * intensity,
                        conflict_importance=conflict_importance * 0.8))
    return out


def extract_from_backend(event: BackendEvent) -> list[Concept]:
    meta = event.metadata or {}
    if meta.get("concept_name"):
        return [_concept(
            str(meta["concept_name"]),
            str(meta.get("concept_value", event.value)),
            "backend",
            str(meta.get("evidence", f"{event.event_type}={event.value}")),
            confidence=float(meta.get("confidence", 0.97)),
            relevance=float(meta.get("relevance", 0.90)),
            conflict_importance=float(meta.get("conflict_importance", 0.0)),
            status=str(meta.get("status", "supported")),
        )]
    return [_concept("backend_event", event.value, "backend", f"{event.event_type}={event.value}", confidence=0.90, relevance=0.55)]


def extract_from_image(obs: ImageObservation) -> list[Concept]:
    if obs.concept_name:
        return [_concept(
            obs.concept_name,
            obs.concept_value or obs.description,
            "image",
            obs.description,
            confidence=obs.confidence,
            relevance=obs.relevance,
            conflict_importance=obs.conflict_importance,
        )]
    return [_concept("visual_observation", obs.description, "image", obs.description,
                     confidence=obs.confidence, relevance=obs.relevance, conflict_importance=obs.conflict_importance)]


def merge_concepts(existing: list[Concept], incoming: Iterable[Concept]) -> list[Concept]:
    by_name = {c.name: c for c in existing}
    for new in incoming:
        old = by_name.get(new.name)
        if old is None:
            existing.append(new)
            by_name[new.name] = new
            continue
        # Backend domain context is authoritative for a generated/manual case; later text cues
        # may be ambiguous (e.g. a fraud case mentioning a card or a device case mentioning Wi-Fi).
        if not (old.name == "customer_domain" and "backend" in old.sources and new.sources == ["text"]):
            old.value = new.value
        old.confidence = max(old.confidence, new.confidence)
        old.task_relevance = max(old.task_relevance, new.task_relevance)
        old.conflict_importance = max(old.conflict_importance, new.conflict_importance)
        old.recency = 1.0
        old.updated_at = _now()
        old.status = new.status
        old.evidence.extend(new.evidence[-2:])
        for src in new.sources:
            if src not in old.sources:
                old.sources.append(src)
    return existing


def decay_recency(concepts: list[Concept], factor: float = 0.92) -> None:
    for c in concepts:
        c.recency = max(0.08, round(c.recency * factor, 4))


def _parse_intensity(by_name: dict[str, Concept]) -> float:
    try:
        return float(by_name.get("emotion_intensity").value) if by_name.get("emotion_intensity") else 0.0
    except ValueError:
        return 0.0


def detect_conflicts(concepts: list[Concept]) -> list[Conflict]:
    by_name = {c.name: c for c in concepts}
    conflicts: list[Conflict] = []
    authoritative = by_name.get("authoritative_status")
    customer_status = by_name.get("customer_belief_status")
    visible = by_name.get("customer_visible_status")

    if authoritative and authoritative.value == "unresolved" and customer_status and customer_status.value == "resolved":
        customer_status.status = authoritative.status = "disputed"
        customer_status.conflict_importance = authoritative.conflict_importance = 1.0
        conflicts.append(Conflict(
            id=_cid("conflict"),
            concept_ids=[customer_status.id, authoritative.id],
            description="The customer believes the issue is resolved, but the authoritative system still shows it as unresolved.",
            description_zh="客户认为问题已经解决，但权威系统仍显示为未解决。",
            severity="high",
            confidence=0.98,
        ))

    if authoritative and authoritative.value == "unresolved" and visible and visible.value.lower() in RESOLVED_VISUAL_VALUES:
        visible.status = authoritative.status = "disputed"
        visible.conflict_importance = max(visible.conflict_importance, 0.96)
        authoritative.conflict_importance = 1.0
        conflicts.append(Conflict(
            id=_cid("conflict"),
            concept_ids=[visible.id, authoritative.id],
            description=f"Customer-facing evidence suggests '{visible.value}', while the authoritative system remains unresolved.",
            description_zh=f"面向客户的证据显示“{CONFLICT_VALUE_ZH.get(visible.value.lower(), visible.value)}”，但权威系统仍显示问题尚未解决。",
            severity="high",
            confidence=0.96,
        ))

    emotion = by_name.get("customer_emotion")
    intensity = _parse_intensity(by_name)
    if customer_status and customer_status.value == "resolved" and emotion and emotion.value in NEGATIVE_EMOTIONS and intensity >= 0.66:
        customer_status.status = emotion.status = "disputed"
        customer_status.conflict_importance = max(customer_status.conflict_importance, 0.72)
        emotion.conflict_importance = max(emotion.conflict_importance, 0.72)
        conflicts.append(Conflict(
            id=_cid("conflict"),
            concept_ids=[customer_status.id, emotion.id],
            description=f"The customer says the issue is resolved, but their {emotion.value} affect remains strong ({intensity:.0%}).",
            description_zh=f"客户表示问题已经解决，但其“{CONFLICT_EMOTION_ZH.get(emotion.value, emotion.value)}”情绪仍然较强（{intensity:.0%}）。",
            severity="medium",
            confidence=min(0.94, 0.62 + intensity * 0.32),
        ))
    return conflicts


def choose_active(concepts: list[Concept], capacity_k: int, preserve_conflicts: bool, conflicts: list[Conflict]) -> list[Concept]:
    ranked = sorted(concepts, key=lambda c: c.score, reverse=True)
    selected = ranked[:capacity_k]
    if preserve_conflicts:
        conflict_ids = {cid for conflict in conflicts for cid in conflict.concept_ids}
        selected_ids = {c.id for c in selected}
        for c in [x for x in ranked if x.id in conflict_ids]:
            if c.id in selected_ids:
                continue
            evictable = [x for x in selected if not x.pinned and x.id not in conflict_ids]
            if evictable:
                victim = min(evictable, key=lambda x: x.score)
                selected.remove(victim)
                selected_ids.remove(victim.id)
            if len(selected) < capacity_k:
                selected.append(c)
                selected_ids.add(c.id)
    return sorted(selected, key=lambda c: c.score, reverse=True)


def recommend_action(state: SessionState) -> tuple[str, str]:
    active = {c.name: c for c in state.active_concepts}
    if state.session_phase == "closing":
        return "close_session", "Acknowledge that there are no other concerns, thank the customer, and end the conversation warmly."
    authoritative = active.get("authoritative_status")
    if authoritative and authoritative.value == "resolved" and not state.conflicts:
        return "confirm_resolution", "Confirm the resolution clearly, then ask whether the customer has any other questions or concerns."

    # Once a root cause is known, move to remediation even if customer-visible evidence
    # still conflicts with the backend.  Prioritizing the conflict forever created an
    # artificial loop where both the customer and agent kept re-verifying the same
    # mismatch instead of fixing the diagnosed cause.
    root = active.get("root_cause")
    domain = active.get("customer_domain")
    if root:
        d = domain.value if domain else "service"
        verbs = {
            "payment": "explain the decline and apply the valid remediation",
            "delivery": "trace, replace, or escalate the shipment",
            "internet": "address the network cause instead of repeating router restarts",
            "account_access": "complete the correct verification or unlock flow",
            "subscription": "correct cancellation and billing state",
            "travel": "repair the ticket/booking and protect the itinerary",
            "return_refund": "restart or escalate the refund workflow",
            "insurance_claim": "identify the missing requirement and advance the claim",
            "device_support": "address the device/firmware failure or replace the device if appropriate",
            "software_saas": "repair the entitlement or permission state",
            "utilities": "correct the billing ledger using the authoritative meter record",
            "healthcare_appointment": "verify scheduling and offer the earliest valid appointment",
            "banking_fraud": "secure the card, explain the dispute state, and progress replacement",
            "hotel_hospitality": "synchronize the reservation with the property system",
            "rideshare": "explain and release the authorization hold or escalate it",
            "event_ticketing": "activate the ticket entitlement before the event",
            "telecom_mobile": "repair network provisioning for the new plan",
            "marketplace_dispute": "complete the outstanding remediation or offer the valid remedy",
        }
        return "act_on_root_cause", f"Use the confirmed root cause — {root.value} — to {verbs.get(d, 'take the next concrete resolution step')}."

    if state.conflicts:
        return "resolve_conflict", "Report the authoritative result clearly, explain the mismatch once, and move toward diagnosis rather than repeatedly re-checking it."

    if active.get("avoid_repeat_action"):
        return "avoid_repetition", "Do not repeat troubleshooting the customer already completed; move to the next diagnostic step."
    if active.get("authoritative_status") and active["authoritative_status"].value == "unresolved":
        return "investigate", "Keep the case open and inspect the authoritative system state before promising resolution."
    return "clarify", "Ask one focused question that advances the issue without making the customer repeat information."


def synthesize_response(state: SessionState) -> str:
    active = {c.name: c for c in state.active_concepts}
    emotion = state.current_emotion or "neutral"
    prefix = {
        "angry": "I can see why this is frustrating.",
        "frustrated": "I can see you've already spent time on this.",
        "impatient": "I'll keep this focused and avoid repeating steps.",
        "anxious": "I checked the current state so I can give you a clear answer.",
        "confused": "The signals you're seeing don't line up, so I'll separate them clearly.",
        "skeptical": "I won't ask you to rely on the screen alone — I checked the authoritative system state.",
        "distressed": "I'll focus on the most immediate next step first.",
        "disappointed": "I understand why this is disappointing.",
        "relieved": "Good — we're closer to a confirmed resolution.",
        "appreciative": "Absolutely.",
    }.get(emotion, "")

    if state.recommended_action_code == "close_session":
        body = "I'm glad we could get this wrapped up. Thanks for contacting support, and I hope you have a great day."
    elif state.recommended_action_code == "confirm_resolution":
        body = "The system now shows the issue as resolved. Is there anything else I can help you with today?"
    elif state.recommended_action_code == "resolve_conflict":
        body = "I checked the authoritative state, and it still conflicts with what appears on your side. I won't call this resolved yet because the backend record is still incomplete."
    elif state.recommended_action_code == "act_on_root_cause" and active.get("root_cause"):
        body = f"I found the underlying issue: {active['root_cause'].value}. That finding explains the symptom, and I can address that specific cause instead of sending you through generic troubleshooting again."
    elif state.recommended_action_code == "avoid_repetition":
        body = "You already completed that troubleshooting, so I won't make you repeat it. I'll move to the next diagnostic step."
    elif state.recommended_action_code == "investigate":
        body = "I checked the authoritative system, and it still shows this as unresolved. The issue is not confirmed fixed yet, so I'll keep the case open rather than give you a false resolution."
    else:
        body = "Tell me the part that matters most right now, and I'll focus on the next useful step."
    return f"{prefix} {body}".strip()


def refresh_state(state: SessionState) -> SessionState:
    state.conflicts = detect_conflicts(state.concepts)
    state.active_concepts = choose_active(
        state.concepts,
        capacity_k=state.config.capacity_k,
        preserve_conflicts=state.config.preserve_conflicts,
        conflicts=state.conflicts,
    )
    by_name = {c.name: c for c in state.concepts}
    emotion = by_name.get("customer_emotion")
    intensity = by_name.get("emotion_intensity")
    if emotion:
        state.current_emotion = emotion.value  # type: ignore[assignment]
    if intensity:
        try:
            state.current_emotion_intensity = float(intensity.value)
        except ValueError:
            pass
    code, text = recommend_action(state)
    state.recommended_action_code = code
    state.recommended_action = text
    state.last_response = synthesize_response(state)
    return state
