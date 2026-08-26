from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Iterable

from .schemas import BackendEvent, Concept, Conflict, CustomerTurn, Evidence, ImageObservation, SessionConfig, SessionState


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
        confidence=confidence,
        task_relevance=relevance,
        conflict_importance=conflict_importance,
        status=status,
    )


def extract_from_turn(turn: CustomerTurn) -> list[Concept]:
    text = turn.text.strip()
    low = text.lower()
    out: list[Concept] = []

    # Generic state / resolution language used across all domains.
    resolved_phrases = ["it worked", "it's working", "its working", "fixed now", "resolved", "all good now",
                        "fine now", "looks fixed", "looks good", "it arrived", "i got it", "can log in now"]
    unresolved_phrases = ["still not working", "still failing", "not fixed", "still broken", "still doesn't work",
                          "still locked", "still missing", "never arrived", "still pending", "not here yet"]
    resolved_signal = any(k in low for k in resolved_phrases) or any(k in low for k in [
        "worked now", "working now", "got back in now", "cancelled now", "canceled now",
        "completed now", "delivered now", "resolved now", "seems resolved", "looks resolved"
    ])
    if resolved_signal:
        out.append(_concept("customer_belief_status", "resolved", "text", text,
                            confidence=0.86, relevance=0.93, conflict_importance=0.55))
    if any(k in low for k in unresolved_phrases):
        out.append(_concept("customer_belief_status", "unresolved", "text", text,
                            confidence=0.92, relevance=0.96, conflict_importance=0.30))

    # Repeated effort is broadly relevant in customer support.
    m = re.search(r"(?:tried|attempted|called|restarted|reset)(?: it)?\s+(\w+|\d+)\s+times", low)
    if m:
        raw = m.group(1)
        word_map = {"once": 1, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        n = word_map.get(raw, int(raw) if raw.isdigit() else None)
        if n:
            out.append(_concept("retry_count", str(n), "text", text, confidence=0.96, relevance=0.93))
            out.append(_concept("avoid_repeat_action", "do not repeat already-attempted troubleshooting", "derived", text,
                                confidence=0.91, relevance=0.98))
    if any(k in low for k in ["already tried", "already did", "already restarted", "asked me that already",
                              "keep trying", "done that already", "already called"]):
        out.append(_concept("troubleshooting_already_attempted", "customer already completed a suggested step", "text", text,
                            confidence=0.90, relevance=0.92))
        out.append(_concept("avoid_repeat_action", "do not repeat already-attempted troubleshooting", "derived", text,
                            confidence=0.91, relevance=0.98))

    # Domain cues. These don't decide the answer; they help keep the active workspace interpretable.
    domain_patterns = {
        "payment": ["payment", "pay", "card", "transaction", "checkout", "declined"],
        "delivery": ["package", "delivery", "shipment", "tracking", "arrive", "courier"],
        "internet": ["internet", "wifi", "wi-fi", "router", "modem", "connection", "offline"],
        "account_access": ["login", "log in", "password", "locked", "account", "verification code"],
        "subscription": ["subscription", "cancel", "membership", "renewal", "charged again"],
        "travel": ["flight", "booking", "reservation", "hotel", "seat", "rebook"],
        "return_refund": ["refund", "return", "returned", "money back"],
        "insurance_claim": ["claim", "insurance", "documents", "adjuster", "coverage"],
    }
    for domain, patterns in domain_patterns.items():
        if any(p in low for p in patterns):
            out.append(_concept("customer_domain", domain, "text", text, confidence=0.92, relevance=1.0, conflict_importance=0.10))
            break

    if any(k in low for k in ["failed", "declined", "not working", "won't go through", "doesn't work", "keeps failing"]):
        out.append(_concept("customer_reported_problem", "operation failing", "text", text,
                            confidence=0.93, relevance=0.94))
    if any(k in low for k in ["late", "missing", "never arrived", "not here"]):
        out.append(_concept("customer_reported_problem", "item or service missing/late", "text", text,
                            confidence=0.92, relevance=0.94))
    if any(k in low for k in ["locked out", "can't log in", "cannot log in", "password doesn't work"]):
        out.append(_concept("customer_reported_problem", "account access blocked", "text", text,
                            confidence=0.94, relevance=0.96))

    if turn.audio_tone in {"frustrated", "angry"}:
        out.append(_concept("customer_sentiment", turn.audio_tone, "audio", f"vocal tone={turn.audio_tone}",
                            confidence=0.78 if turn.audio_tone == "frustrated" else 0.86,
                            relevance=0.78, conflict_importance=0.25))
    elif turn.audio_tone == "uncertain":
        out.append(_concept("customer_sentiment", "uncertain", "audio", "vocal tone=uncertain",
                            confidence=0.76, relevance=0.72, conflict_importance=0.20))
    elif turn.audio_tone in {"calm", "neutral"}:
        out.append(_concept("customer_sentiment", turn.audio_tone, "audio", f"vocal tone={turn.audio_tone}",
                            confidence=0.72, relevance=0.55))

    return out


def extract_from_backend(event: BackendEvent) -> list[Concept]:
    out: list[Concept] = []
    meta = event.metadata or {}

    # Generic structured event used by the automated simulator and future real CRM adapters.
    if meta.get("concept_name"):
        name = str(meta["concept_name"])
        value = str(meta.get("concept_value", event.value))
        out.append(_concept(
            name,
            value,
            "backend",
            str(meta.get("evidence", f"{event.event_type}={event.value}")),
            confidence=float(meta.get("confidence", 0.98)),
            relevance=float(meta.get("relevance", 0.90)),
            conflict_importance=float(meta.get("conflict_importance", 0.0)),
            status=str(meta.get("status", "supported")),
        ))
        return out

    # Backward-compatible payment events from v0.1.
    if event.event_type == "payment_attempt":
        out.append(_concept("backend_attempt_count", str(event.value), "backend", f"payment_attempt={event.value}",
                            confidence=0.99, relevance=0.95))
    elif event.event_type == "payment_status":
        value = event.value.lower()
        out.append(_concept("payment_status", value, "backend", f"payment_status={event.value}",
                            confidence=0.995, relevance=0.99, conflict_importance=0.65))
        status = "unresolved" if value in {"failed", "declined"} else "resolved" if value in {"succeeded", "success", "approved"} else value
        out.append(_concept("authoritative_status", status, "backend", f"backend payment status={event.value}",
                            confidence=0.99, relevance=0.99, conflict_importance=0.70))
    elif event.event_type == "decline_reason":
        out.append(_concept("root_cause", event.value, "backend", f"decline_reason={event.value}",
                            confidence=0.99, relevance=1.0))
    elif event.event_type == "case_status":
        out.append(_concept("case_status", event.value, "backend", f"case_status={event.value}",
                            confidence=0.98, relevance=0.88))
    else:
        out.append(_concept("backend_event", event.value, "backend", str(event.metadata) or event.value,
                            confidence=0.90, relevance=0.60))
    return out


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

    low = obs.description.lower()
    out: list[Concept] = []
    error = re.search(r"(?:error|code)\s*([a-z0-9-]+)", low)
    if error:
        out.append(_concept("visual_error_code", error.group(1), "image", obs.description,
                            confidence=0.92, relevance=0.88))
    if any(k in low for k in ["failed", "declined", "unsuccessful", "error", "offline", "locked", "cancelled"]):
        out.append(_concept("visual_problem_evidence", obs.description, "image", obs.description,
                            confidence=0.82, relevance=0.86))
    if not out:
        out.append(_concept("visual_observation", obs.description, "image", obs.description,
                            confidence=0.70, relevance=0.50))
    return out


def merge_concepts(existing: list[Concept], incoming: Iterable[Concept]) -> list[Concept]:
    by_name = {c.name: c for c in existing}
    for new in incoming:
        old = by_name.get(new.name)
        if old is None:
            existing.append(new)
            by_name[new.name] = new
            continue
        old.value = new.value
        old.confidence = max(old.confidence, new.confidence)
        old.task_relevance = max(old.task_relevance, new.task_relevance)
        old.conflict_importance = max(old.conflict_importance, new.conflict_importance)
        old.recency = 1.0
        old.updated_at = _now()
        old.status = new.status
        old.evidence.extend(new.evidence)
        for src in new.sources:
            if src not in old.sources:
                old.sources.append(src)
    return existing


def decay_recency(concepts: list[Concept], factor: float = 0.94) -> None:
    for c in concepts:
        c.recency = max(0.1, round(c.recency * factor, 4))


def detect_conflicts(concepts: list[Concept]) -> list[Conflict]:
    by_name = {c.name: c for c in concepts}
    conflicts: list[Conflict] = []

    customer_status = by_name.get("customer_belief_status")
    authoritative = by_name.get("authoritative_status")
    if customer_status and authoritative and customer_status.value != authoritative.value:
        customer_status.status = "disputed"
        authoritative.status = "disputed"
        customer_status.conflict_importance = 1.0
        authoritative.conflict_importance = 1.0
        conflicts.append(Conflict(
            id=_cid("conflict"),
            concept_ids=[customer_status.id, authoritative.id],
            description=(f"Customer believes the issue is {customer_status.value}, but the authoritative system says "
                         f"{authoritative.value}."),
            severity="high",
            confidence=0.98,
        ))

    sentiment = by_name.get("customer_sentiment")
    if customer_status and sentiment and customer_status.value == "resolved" and sentiment.value in {"frustrated", "angry"}:
        customer_status.status = "disputed"
        sentiment.status = "disputed"
        customer_status.conflict_importance = max(customer_status.conflict_importance, 0.75)
        sentiment.conflict_importance = max(sentiment.conflict_importance, 0.75)
        conflicts.append(Conflict(
            id=_cid("conflict"),
            concept_ids=[customer_status.id, sentiment.id],
            description="Customer says the issue is resolved while vocal cues remain negative.",
            severity="medium",
            confidence=0.80,
        ))

    # Generic explicit pair conflict: scenario/backend adapters can publish two concepts with matching conflict_group.
    # Pair-specific handling can be added later; the status conflict above is the core research case.
    return conflicts


def choose_active(concepts: list[Concept], config: SessionConfig, conflicts: list[Conflict]) -> list[Concept]:
    ranked = sorted(concepts, key=lambda c: c.score, reverse=True)
    selected = ranked[: config.capacity_k]

    if config.preserve_conflicts:
        conflict_ids = {cid for conflict in conflicts for cid in conflict.concept_ids}
        required = [c for c in ranked if c.id in conflict_ids]
        selected_ids = {c.id for c in selected}
        for c in required:
            if c.id in selected_ids:
                continue
            evictable = [x for x in selected if not x.pinned and x.id not in conflict_ids]
            if evictable:
                victim = sorted(evictable, key=lambda x: x.score)[0]
                selected.remove(victim)
                selected_ids.remove(victim.id)
            if len(selected) < config.capacity_k:
                selected.append(c)
                selected_ids.add(c.id)

    return sorted(selected, key=lambda c: c.score, reverse=True)


def recommend_action(state: SessionState) -> tuple[str, str]:
    active = {c.name: c for c in state.active_concepts}
    if state.conflicts:
        return "resolve_authoritative_conflict", "Verify the authoritative system state and explain the mismatch before closing the issue."

    # Domain-specific operational actions.
    if active.get("root_cause"):
        domain = active.get("customer_domain")
        d = domain.value if domain else ""
        cause = active["root_cause"].value
        if d == "payment":
            return "explain_payment_decline", f"Explain the payment failure reason and the next valid remediation: {cause}."
        if d == "delivery":
            return "trace_or_replace_shipment", f"Use the confirmed delivery cause to trace, replace, or escalate the shipment: {cause}."
        if d == "internet":
            return "address_network_cause", f"Address the confirmed network cause instead of repeating generic restarts: {cause}."
        if d == "account_access":
            return "restore_account_access", f"Resolve the account-access cause using the appropriate verification/unlock flow: {cause}."
        if d == "subscription":
            return "fix_subscription_state", f"Correct the subscription/cancellation state and explain any billing impact: {cause}."
        if d == "travel":
            return "rebook_or_offer_alternative", f"Use the booking cause to rebook or offer the next valid alternative: {cause}."
        if d == "return_refund":
            return "resolve_refund", f"Resolve the refund/return state based on the confirmed cause: {cause}."
        if d == "insurance_claim":
            return "advance_claim", f"Explain what is blocking the claim and the exact next step: {cause}."
        return "resolve_root_cause", f"Act on the confirmed root cause: {cause}."

    status = active.get("authoritative_status")
    if status and status.value == "unresolved":
        domain = active.get("customer_domain")
        d = domain.value if domain else ""
        codes = {
            "payment": ("inspect_payment_failure", "Inspect the payment failure reason before asking for another attempt."),
            "delivery": ("trace_shipment", "Trace the shipment using carrier/warehouse evidence before promising delivery."),
            "internet": ("diagnose_network", "Check outage and line/device status before asking for another restart."),
            "account_access": ("diagnose_access", "Check lock, authentication, and verification state before another password reset."),
            "subscription": ("inspect_subscription", "Check cancellation and billing state before confirming the subscription is closed."),
            "travel": ("inspect_booking", "Check authoritative booking status and available alternatives."),
            "return_refund": ("inspect_refund", "Check return receipt and refund processing state."),
            "insurance_claim": ("inspect_claim", "Check claim status and outstanding requirements."),
        }
        return codes.get(d, ("investigate_unresolved", "Investigate the authoritative unresolved state before repeating troubleshooting."))

    if active.get("avoid_repeat_action"):
        return "avoid_repetition", "Do not repeat already-completed troubleshooting; choose the next diagnostic or escalation step."

    if status and status.value == "resolved":
        return "confirm_resolution", "Confirm the authoritative resolution and close with a concise check that the customer is satisfied."

    return "clarify_goal", "Ask one concise question to identify the customer's current goal and issue."


def synthesize_response(state: SessionState) -> str:
    code = state.recommended_action_code
    active = {c.name: c for c in state.active_concepts}
    if code == "resolve_authoritative_conflict":
        auth = active.get("authoritative_status")
        if auth:
            return (f"I’m seeing a mismatch: it may look resolved on your side, but our system still shows it as {auth.value}. "
                    "I won’t close this yet; I’ll verify the authoritative status and take the next step from there.")
        return "I’m seeing conflicting signals, so I’m going to verify the authoritative status before I give you the next step."
    if code == "avoid_repetition":
        return "I can see you already tried that, so I won’t make you repeat it. I’ll move to the next diagnostic step."
    if code == "confirm_resolution":
        return "Our system now shows the issue as resolved. Before I close this, can you confirm everything is working correctly on your side?"
    if active.get("root_cause"):
        return f"I found what is blocking this: {active['root_cause'].value}. I’ll use that to take the next valid step instead of repeating generic troubleshooting."
    if active.get("authoritative_status") and active["authoritative_status"].value == "unresolved":
        return "I can confirm the issue is still unresolved in our system. I’ll check the specific cause and next action rather than asking you to repeat what you’ve already tried."
    return "I’m here to help. Tell me what happened, and I’ll work through the current state with you."


def refresh_state(state: SessionState) -> SessionState:
    state.conflicts = detect_conflicts(state.concepts)
    state.active_concepts = choose_active(state.concepts, state.config, state.conflicts)
    code, text = recommend_action(state)
    state.recommended_action_code = code
    state.recommended_action = text
    state.last_response = synthesize_response(state)
    return state
