from __future__ import annotations

import random
import uuid

from .schemas import BackendEvent, CustomerTurn, GeneratedScenario, ImageObservation, ScenarioControls, ScenarioStep


def _structured_event(event_type: str, concept_name: str, value: str, *, evidence: str,
                      relevance: float = 0.95, confidence: float = 0.98,
                      conflict_importance: float = 0.0) -> BackendEvent:
    return BackendEvent(
        event_type=event_type,
        value=value,
        metadata={
            "concept_name": concept_name,
            "concept_value": value,
            "evidence": evidence,
            "relevance": relevance,
            "confidence": confidence,
            "conflict_importance": conflict_importance,
        },
    )


def _noise_events(rng: random.Random, n: int) -> list[BackendEvent]:
    options = [
        ("loyalty_tier", "silver"), ("marketing_opt_in", "true"), ("profile_language", "English"),
        ("last_login_region", "New York"), ("account_age", "3 years"), ("preferred_channel", "email"),
        ("promo_eligibility", "eligible"), ("profile_complete", "true"),
    ]
    rng.shuffle(options)
    return [
        _structured_event("context", f"noise_{name}", value, evidence=f"non-critical context: {name}={value}",
                          relevance=0.12, confidence=0.95)
        for name, value in options[:n]
    ]


def _profile(rng: random.Random) -> dict:
    return {
        "persona": rng.choice(["patient", "busy professional", "detail-oriented", "impatient", "first-time user"]),
        "channel": "voice + chat",
        "tenure": rng.choice(["new", "6 months", "2 years", "5 years"]),
    }


def _conflict_turn(domain: str) -> str:
    variants = {
        "payment": "Okay, I think the payment finally worked now. It's fine.",
        "delivery": "The app says delivered now, so I guess it is resolved.",
        "internet": "The Wi-Fi icon came back, so I think it's fixed now.",
        "account_access": "I think I got back in now, so it should be fine.",
        "subscription": "The page says cancelled now, so I guess I'm done.",
        "travel": "It looks like the change went through, so I think the booking is resolved.",
        "return_refund": "The app says completed now, so I assume the refund is resolved.",
        "insurance_claim": "The portal looks updated, so I think the claim is all good now.",
    }
    return variants[domain]


def _blueprints() -> dict[str, dict]:
    return {
        "payment": {
            "title": "Repeated payment failure",
            "opening": "My card keeps getting declined at checkout. I've tried it three times and it's still not working.",
            "tone": "frustrated",
            "status_event": ("payment_status", "unresolved", "processor shows the latest authorization failed"),
            "root_cause": "merchant category restriction",
            "root_cause_evidence": "issuer decline code maps to merchant category restriction",
            "image": ("Checkout screenshot shows 'Transaction unsuccessful'", "visual_problem_evidence", "transaction unsuccessful"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "delivery": {
            "title": "Package marked delivered but missing",
            "opening": "My package still hasn't arrived. The tracking page keeps changing and it's not here yet.",
            "tone": "frustrated",
            "status_event": ("shipment_status", "unresolved", "carrier scan shows package held at local depot, not delivered"),
            "root_cause": "carrier depot exception",
            "root_cause_evidence": "carrier API reports depot exception after failed route scan",
            "image": ("Tracking screenshot displays a green 'Delivered' badge", "customer_visible_status", "delivered"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "internet": {
            "title": "Recurring home internet outage",
            "opening": "My internet keeps dropping. I already restarted the router three times and it's still not working.",
            "tone": "angry",
            "status_event": ("network_status", "unresolved", "line monitor shows active neighborhood outage"),
            "root_cause": "neighborhood fiber outage",
            "root_cause_evidence": "network operations system reports fiber incident affecting the node",
            "image": ("Router photo shows a red WAN light", "device_indicator", "red WAN light"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "account_access": {
            "title": "Account login lockout",
            "opening": "I can't log in. I already reset my password three times and I'm still locked out.",
            "tone": "frustrated",
            "status_event": ("account_status", "unresolved", "identity service shows account security lock remains active"),
            "root_cause": "risk lock requires identity verification",
            "root_cause_evidence": "identity platform shows risk-based lock requiring verification",
            "image": ("Login screenshot says 'Too many attempts — account locked'", "visual_problem_evidence", "account locked"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "subscription": {
            "title": "Cancellation did not take effect",
            "opening": "I cancelled my subscription last week but I was charged again. I already went through the cancellation flow.",
            "tone": "frustrated",
            "status_event": ("subscription_status", "unresolved", "billing system shows subscription still active"),
            "root_cause": "cancellation confirmation was never committed",
            "root_cause_evidence": "billing audit shows checkout exit before final cancellation commit",
            "image": ("Account page screenshot shows 'Cancellation requested'", "customer_visible_status", "cancellation requested"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "travel": {
            "title": "Flight change appears successful but is not ticketed",
            "opening": "I changed my flight in the app, but I'm not sure it actually went through and I need to travel tomorrow.",
            "tone": "uncertain",
            "status_event": ("booking_status", "unresolved", "reservation changed but ticket reissue failed"),
            "root_cause": "ticket reissue failed after itinerary change",
            "root_cause_evidence": "reservation system has new itinerary but old ticket coupon remains active",
            "image": ("App screenshot shows the new flight under 'Upcoming trips'", "customer_visible_status", "new itinerary visible"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "return_refund": {
            "title": "Returned item but refund still pending",
            "opening": "I returned the item two weeks ago and I still haven't received my refund. I've already contacted support twice.",
            "tone": "frustrated",
            "status_event": ("refund_status", "unresolved", "warehouse received return but refund job is pending"),
            "root_cause": "refund workflow stalled after warehouse receipt",
            "root_cause_evidence": "return received event exists but finance refund task was not created",
            "image": ("Return portal screenshot says 'Return completed'", "customer_visible_status", "return completed"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
        "insurance_claim": {
            "title": "Claim appears complete but is blocked",
            "opening": "My claim has been sitting there for days. The portal looks complete but nobody can tell me why it hasn't moved.",
            "tone": "frustrated",
            "status_event": ("claim_status", "unresolved", "claim workflow is blocked awaiting one required document"),
            "root_cause": "proof-of-loss document missing",
            "root_cause_evidence": "claims system marks proof-of-loss as mandatory and absent",
            "image": ("Claim portal screenshot shows a 100% progress bar", "customer_visible_status", "100% progress"),
            "expected_action_code": "resolve_authoritative_conflict",
            "critical": ["authoritative_status", "customer_belief_status"],
        },
    }


def list_domains() -> list[str]:
    return sorted(_blueprints().keys())


def generate_scenario(controls: ScenarioControls) -> GeneratedScenario:
    seed = controls.seed if controls.seed is not None else random.SystemRandom().randint(1, 2_000_000_000)
    rng = random.Random(seed)
    blueprints = _blueprints()
    domain = controls.domain if controls.domain in blueprints else rng.choice(list(blueprints))
    b = blueprints[domain]

    include_conflict = controls.include_conflict
    if include_conflict is None:
        include_conflict = controls.difficulty != "easy" or rng.random() < 0.5

    noise_count = {"easy": 0, "medium": 2, "hard": 5}[controls.difficulty]
    status_type, status_value, status_evidence = b["status_event"]

    steps: list[ScenarioStep] = [
        ScenarioStep(
            label="Customer reports the problem",
            customer_turn=CustomerTurn(text=b["opening"], audio_tone=b["tone"]),
            backend_events=_noise_events(rng, noise_count),
        ),
        ScenarioStep(
            label="Authoritative system state arrives",
            backend_events=[
                _structured_event(status_type, "authoritative_status", status_value,
                                  evidence=status_evidence, relevance=0.99, confidence=0.995,
                                  conflict_importance=0.75),
            ],
            image_observations=[ImageObservation(
                description=b["image"][0], concept_name=b["image"][1], concept_value=b["image"][2],
                confidence=0.86, relevance=0.66,
            )],
        ),
        ScenarioStep(
            label="Root cause becomes available",
            backend_events=[
                _structured_event("diagnostic", "root_cause", b["root_cause"], evidence=b["root_cause_evidence"],
                                  relevance=1.0, confidence=0.99),
            ],
        ),
    ]

    expected_action_code = b["expected_action_code"] if include_conflict else {
        "payment": "explain_payment_decline",
        "delivery": "trace_or_replace_shipment",
        "internet": "address_network_cause",
        "account_access": "restore_account_access",
        "subscription": "fix_subscription_state",
        "travel": "rebook_or_offer_alternative",
        "return_refund": "resolve_refund",
        "insurance_claim": "advance_claim",
    }[domain]
    critical = list(b["critical"] if include_conflict else ["root_cause", "authoritative_status"])

    if include_conflict:
        # The customer says/infers it is resolved even though authoritative status remains unresolved.
        steps.append(ScenarioStep(
            label="Customer reports apparent resolution",
            customer_turn=CustomerTurn(text=_conflict_turn(domain), audio_tone=rng.choice(["neutral", "uncertain", "frustrated"])),
        ))

    return GeneratedScenario(
        scenario_id=f"scn_{uuid.uuid4().hex[:10]}",
        domain=domain,
        title=b["title"],
        difficulty=controls.difficulty,
        customer_profile=_profile(rng),
        hidden_ground_truth={
            "authoritative_status": status_value,
            "root_cause": b["root_cause"],
            "customer_visible_evidence": b["image"][2],
            "conflict_injected": include_conflict,
        },
        expected_action_code=expected_action_code,
        expected_conflict=bool(include_conflict),
        critical_concepts=critical,
        steps=steps,
        seed=seed,
    )
