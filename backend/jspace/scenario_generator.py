from __future__ import annotations

import random
import uuid

from .schemas import BackendEvent, CustomerTurn, GeneratedScenario, ImageObservation, ScenarioControls, ScenarioStep


EMOTIONS = [
    "calm", "neutral", "curious", "hopeful", "appreciative", "satisfied", "relieved",
    "uncertain", "confused", "anxious", "disappointed", "frustrated", "angry", "impatient",
    "skeptical", "distressed", "embarrassed",
]

NEGATIVE_EMOTIONS = ["uncertain", "confused", "anxious", "disappointed", "frustrated", "angry", "impatient", "skeptical", "distressed"]
POSITIVE_EMOTIONS = ["calm", "hopeful", "appreciative", "satisfied", "relieved"]


def _event(event_type: str, concept_name: str, value: str, *, evidence: str,
           relevance: float = 0.95, confidence: float = 0.97,
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


def _blueprints() -> dict[str, dict]:
    return {
        "payment": {
            "title": "Repeated payment failure",
            "opening": "My card keeps getting declined at checkout. I've already tried a few times.",
            "impact": "I need this purchase to go through today. Is there something wrong with my account?",
            "followup": "Please don't just tell me to retry again — I already did that.",
            "apparent": "The screen looks like it went through now. Is it actually successful?",
            "final": "Can you tell me exactly what happens next so I don't get charged twice?",
            "status_event": ("payment_status", "unresolved", "processor shows the latest authorization failed"),
            "root_cause": "merchant category restriction",
            "root_evidence": "issuer decline code maps to a merchant category restriction",
            "visual": "Checkout screen briefly shows a success checkmark even though authorization failed",
            "visible_value": "appears successful",
        },
        "delivery": {
            "title": "Package status mismatch",
            "opening": "My order still isn't here, but the tracking page keeps changing.",
            "impact": "It's a gift I need tonight. Can you see where the package actually is?",
            "followup": "I've checked the lobby and asked my neighbors already.",
            "apparent": "Now the app says delivered. Does that mean someone actually dropped it off?",
            "final": "If it's not coming today, what can you do instead?",
            "status_event": ("shipment_status", "unresolved", "carrier scan shows the parcel held at the local depot"),
            "root_cause": "carrier depot exception",
            "root_evidence": "carrier API reports a depot exception after a failed route scan",
            "visual": "Tracking page displays a green Delivered badge",
            "visible_value": "delivered",
        },
        "internet": {
            "title": "Recurring home internet outage",
            "opening": "My internet keeps dropping and I've already restarted the router multiple times.",
            "impact": "I'm working from home and I have a call soon. Is this my router or your network?",
            "followup": "The lights keep changing, so I don't know what I'm supposed to trust.",
            "apparent": "The Wi-Fi icon is back. Is the outage actually over?",
            "final": "What should I do if it drops again in ten minutes?",
            "status_event": ("network_status", "unresolved", "line monitor shows an active neighborhood outage"),
            "root_cause": "neighborhood fiber outage",
            "root_evidence": "network operations reports a fiber incident affecting the local node",
            "visual": "Router shows a red WAN indicator while the phone still displays Wi-Fi bars",
            "visible_value": "wifi visible",
        },
        "account_access": {
            "title": "Account login lockout",
            "opening": "I can't get into my account even after resetting my password.",
            "impact": "I need access before a deadline today. Why am I still locked out?",
            "followup": "I've already done the password reset flow more than once.",
            "apparent": "I got past the first screen now. Am I actually unlocked?",
            "final": "Can we make sure I won't get locked out again right after this?",
            "status_event": ("account_status", "unresolved", "identity service shows a security lock remains active"),
            "root_cause": "risk lock requires identity verification",
            "root_evidence": "identity platform shows a risk-based lock requiring verification",
            "visual": "Login UI opens the dashboard shell but still shows a restricted-session banner",
            "visible_value": "partial access",
        },
        "subscription": {
            "title": "Cancellation did not take effect",
            "opening": "I cancelled last week, but I was charged again today.",
            "impact": "I don't want another renewal. Can you confirm whether this is actually cancelled?",
            "followup": "I already went through the cancellation page and got a confirmation screen.",
            "apparent": "The page now says cancelled. Is billing definitely stopped?",
            "final": "And what happens to the charge that already went through?",
            "status_event": ("subscription_status", "unresolved", "billing system shows the subscription remains active"),
            "root_cause": "cancellation confirmation was never committed",
            "root_evidence": "billing audit shows the flow exited before the final cancellation commit",
            "visual": "Account page shows Cancellation requested",
            "visible_value": "cancellation requested",
        },
        "travel": {
            "title": "Flight change not ticketed",
            "opening": "I changed my flight in the app, but I'm not sure the new itinerary is actually ticketed.",
            "impact": "I fly tomorrow morning. Can you confirm I really have a seat on the new flight?",
            "followup": "The itinerary changed, but I never received a new ticket email.",
            "apparent": "The new flight is showing under Upcoming Trips. Does that mean I'm safe to travel?",
            "final": "If the ticket isn't valid, can you fix it without changing my flight again?",
            "status_event": ("booking_status", "unresolved", "reservation changed but ticket reissue failed"),
            "root_cause": "ticket reissue failed after itinerary change",
            "root_evidence": "reservation has the new itinerary while the old ticket coupon remains active",
            "visual": "App shows the new flight under Upcoming Trips",
            "visible_value": "new itinerary visible",
        },
        "return_refund": {
            "title": "Return received but refund stalled",
            "opening": "I sent the item back almost two weeks ago and still don't have my refund.",
            "impact": "The return page says you're done with it. Why is my money still missing?",
            "followup": "I've already contacted support about this once before.",
            "apparent": "Now it says Return completed. Does that mean the refund was sent?",
            "final": "What date should I actually expect the money back?",
            "status_event": ("refund_status", "unresolved", "warehouse received the return but the refund job is pending"),
            "root_cause": "refund workflow stalled after warehouse receipt",
            "root_evidence": "return receipt exists but the finance refund task was not created",
            "visual": "Return portal displays Return completed",
            "visible_value": "return completed",
        },
        "insurance_claim": {
            "title": "Insurance claim blocked by missing requirement",
            "opening": "My claim has looked complete for days, but it still isn't moving.",
            "impact": "I need to know whether you're waiting on me or whether it's stuck internally.",
            "followup": "The portal doesn't show anything else I need to upload.",
            "apparent": "The progress bar says 100%. Is the claim actually complete?",
            "final": "Please tell me the exact missing item so I can resolve this today.",
            "status_event": ("claim_status", "unresolved", "claim workflow is blocked awaiting a required document"),
            "root_cause": "proof-of-loss document missing",
            "root_evidence": "claims system marks proof-of-loss as mandatory and absent",
            "visual": "Claim portal shows a 100% progress bar",
            "visible_value": "100% progress",
        },
        "device_support": {
            "title": "Smart device intermittently failing",
            "opening": "My new smart speaker keeps disconnecting even though setup says it completed.",
            "impact": "It works for a minute and then disappears. Is the device defective?",
            "followup": "I've reset it and re-added it to Wi-Fi already.",
            "apparent": "The app says Connected now. Is the device actually stable?",
            "final": "If it disconnects again, should I replace it or keep troubleshooting?",
            "status_event": ("device_status", "unresolved", "device telemetry shows repeated authentication drops"),
            "root_cause": "firmware authentication loop",
            "root_evidence": "telemetry shows repeated token refresh failures after reconnect",
            "visual": "Device app displays Connected while telemetry is offline",
            "visible_value": "connected",
        },
        "software_saas": {
            "title": "Enterprise software access issue",
            "opening": "The dashboard loads, but I still can't access the workspace my team invited me to.",
            "impact": "I have a client review coming up. Why does it look active if I don't have access?",
            "followup": "My admin already resent the invitation and checked my email address.",
            "apparent": "I can see the workspace name now. Does that mean my permissions are fixed?",
            "final": "Can you confirm which permission is still missing?",
            "status_event": ("workspace_status", "unresolved", "entitlement service shows workspace permission not granted"),
            "root_cause": "role entitlement propagation failed",
            "root_evidence": "identity sync completed but role entitlement job failed",
            "visual": "Workspace appears in navigation but opens an Access denied panel",
            "visible_value": "workspace visible",
        },
        "utilities": {
            "title": "Utility bill adjustment mismatch",
            "opening": "My electricity bill jumped after I was told a meter correction had been applied.",
            "impact": "The app says adjusted, but the amount is still much higher than normal.",
            "followup": "I already submitted the meter photo you requested.",
            "apparent": "The bill now has an Adjusted label. Is the balance final?",
            "final": "Can you explain which reading you're actually billing me for?",
            "status_event": ("billing_status", "unresolved", "billing ledger has not posted the corrected meter read"),
            "root_cause": "corrected meter read awaiting ledger posting",
            "root_evidence": "meter service accepted the correction but billing ledger is still on the prior read",
            "visual": "Bill page shows an Adjusted badge next to the old balance",
            "visible_value": "adjusted badge",
        },
        "healthcare_appointment": {
            "title": "Appointment confirmation mismatch",
            "opening": "The patient portal says my appointment is confirmed, but I got a voicemail saying it may have been cancelled.",
            "impact": "I arranged my schedule around this visit. Do I actually have an appointment?",
            "followup": "I don't want to show up and find out the clinician isn't there.",
            "apparent": "The portal still says Confirmed. Can I rely on that?",
            "final": "If it was cancelled, what is the earliest replacement slot?",
            "status_event": ("appointment_status", "unresolved", "scheduling system shows the clinician block was cancelled"),
            "root_cause": "provider schedule change not reflected in portal cache",
            "root_evidence": "scheduler cancelled the block while the portal cache retained confirmation",
            "visual": "Patient portal still shows a green Confirmed badge",
            "visible_value": "confirmed",
        },
        "banking_fraud": {
            "title": "Fraud alert and card-status confusion",
            "opening": "I got a fraud alert, but the banking app still shows my card as active.",
            "impact": "I don't recognize the purchase. Can the card still be used right now?",
            "followup": "I already marked the transaction as not mine in the app.",
            "apparent": "The card page still says Active. Does that mean it wasn't blocked?",
            "final": "What should I expect for the disputed charge and replacement card?",
            "status_event": ("card_status", "unresolved", "fraud system placed a restricted authorization state on the card"),
            "root_cause": "fraud restriction pending card replacement workflow",
            "root_evidence": "fraud case is open and authorization restrictions are active",
            "visual": "Banking app card tile still displays Active",
            "visible_value": "active",
        },
        "hotel_hospitality": {
            "title": "Hotel reservation room-type mismatch",
            "opening": "My hotel app shows the room upgrade I paid for, but the confirmation email still has the old room type.",
            "impact": "I'm checking in tonight. Which room is actually reserved for me?",
            "followup": "I don't want to arrive and have to argue about the upgrade at the desk.",
            "apparent": "The app now displays the upgraded room. Is that guaranteed?",
            "final": "Can you make sure the hotel property itself sees the same reservation?",
            "status_event": ("reservation_status", "unresolved", "property management system still holds the original room type"),
            "root_cause": "upgrade failed to sync to property system",
            "root_evidence": "central reservation updated but property management sync failed",
            "visual": "Hotel app displays the upgraded room category",
            "visible_value": "upgrade visible",
        },
        "rideshare": {
            "title": "Ride charge after driver cancellation",
            "opening": "The driver cancelled, but I still see a charge for the ride.",
            "impact": "I had to book another car. Am I being charged for both rides?",
            "followup": "The first ride never even picked me up.",
            "apparent": "The app now says Cancelled. Does that automatically remove the charge?",
            "final": "When will the pending amount disappear from my card?",
            "status_event": ("ride_charge_status", "unresolved", "payment ledger still has an authorization hold from the cancelled ride"),
            "root_cause": "authorization hold awaiting automatic release",
            "root_evidence": "ride was cancelled but payment authorization has not yet been released",
            "visual": "Trip screen displays Cancelled while wallet shows a pending charge",
            "visible_value": "cancelled",
        },
        "event_ticketing": {
            "title": "Transferred event ticket not activated",
            "opening": "I transferred my concert ticket, and the recipient can see it but can't open the barcode.",
            "impact": "The event is tonight. Does the recipient actually own the ticket now?",
            "followup": "They already accepted the transfer email.",
            "apparent": "Their account now shows the ticket. Is the transfer complete?",
            "final": "What do we need to do before we get to the venue?",
            "status_event": ("ticket_status", "unresolved", "ticketing service shows transfer accepted but entitlement activation failed"),
            "root_cause": "ticket entitlement activation failed",
            "root_evidence": "transfer record exists but barcode entitlement was not activated",
            "visual": "Recipient account shows the event tile without an active barcode",
            "visible_value": "ticket visible",
        },
        "telecom_mobile": {
            "title": "Mobile plan change not provisioned",
            "opening": "I upgraded my mobile plan, but my phone still behaves like I'm on the old data limit.",
            "impact": "The app shows the new plan. Is the network actually using it?",
            "followup": "I've already restarted the phone and reset network settings.",
            "apparent": "The account page says Upgrade complete. Does that mean provisioning is finished?",
            "final": "Can you make sure I won't get throttled under the old limit?",
            "status_event": ("plan_status", "unresolved", "network provisioning still has the old policy profile"),
            "root_cause": "plan change not propagated to network policy",
            "root_evidence": "billing plan changed but network policy provisioning job failed",
            "visual": "Carrier app displays the upgraded plan name",
            "visible_value": "upgrade complete",
        },
        "marketplace_dispute": {
            "title": "Marketplace dispute appears closed",
            "opening": "The marketplace says my dispute is closed, but the seller never sent the replacement they promised.",
            "impact": "Does Closed mean I lost the case, or is something still being processed?",
            "followup": "I uploaded the photos and messages already.",
            "apparent": "The case page says Completed now. Is there still a replacement coming?",
            "final": "If not, what remedy is actually available to me?",
            "status_event": ("dispute_status", "unresolved", "case system shows remediation task still pending"),
            "root_cause": "seller remediation task was never fulfilled",
            "root_evidence": "dispute decision completed but replacement fulfillment remains open",
            "visual": "Case page shows Completed while fulfillment section remains blank",
            "visible_value": "completed",
        },
    }


def list_domains() -> list[str]:
    return sorted(_blueprints().keys())


def get_blueprint(domain: str) -> dict:
    return _blueprints()[domain]


def _profile(rng: random.Random, domain: str | None = None) -> dict:
    tenure_years = rng.choice([0.1, 0.5, 1, 2, 3, 5, 7, 10, 12])
    relationship = rng.choices(
        ["new", "positive", "loyal", "neutral", "strained", "at risk"],
        weights=[8, 22, 18, 22, 20, 10],
        k=1,
    )[0]
    previous_contacts = rng.randint(0, 5)
    communication_style = rng.choice(["concise", "detail-oriented", "conversational", "direct", "cautious", "question-heavy"])
    # Starting patience reflects customer context rather than always beginning at 100.
    # Most customers still begin fairly patient, while strained/at-risk relationships,
    # repeated recent contacts, and urgent domains can start materially lower.
    relationship_base = {
        "new": 88, "positive": 94, "loyal": 97, "neutral": 86, "strained": 67, "at risk": 52,
    }[relationship]
    urgency_penalty = 8 if domain in {"banking_fraud", "account_access", "travel", "event_ticketing"} else (4 if domain in {"payment", "internet", "utilities"} else 0)
    style_adjustment = {"direct": -4, "question-heavy": -2, "cautious": -1, "concise": 1, "detail-oriented": 0, "conversational": 2}[communication_style]
    patience = int(max(28, min(100, relationship_base - previous_contacts * 4 - urgency_penalty + style_adjustment + rng.randint(-4, 5))))
    trust = rng.randint(45, 88)
    return {
        "name": rng.choice(["Alex", "Maya", "Jordan", "Sam", "Taylor", "Chris", "Morgan", "Riley", "Jamie", "Avery"]),
        "tenure": "new customer" if tenure_years < 0.5 else ((("1 year" if int(tenure_years) == 1 else f"{int(tenure_years)} years") if tenure_years >= 1 else "6 months")),
        "relationship": relationship,
        "loyalty_tier": rng.choice(["standard", "silver", "gold", "platinum"]),
        "previous_contacts_90d": previous_contacts,
        "value_segment": rng.choice(["standard", "high value", "strategic", "occasional"]),
        "communication_style": communication_style,
        "tech_comfort": rng.choice(["low", "medium", "high"]),
        "patience": patience,
        "trust": trust,
        "preferred_channel": rng.choice(["voice", "chat", "mobile app", "email"]),
    }


def _initial_emotion(rng: random.Random, profile: dict) -> tuple[str, float]:
    relationship = profile["relationship"]
    if relationship in {"strained", "at risk"}:
        choices = ["frustrated", "angry", "skeptical", "disappointed", "impatient"]
        lo, hi = 0.62, 0.96
    elif profile["patience"] < 45:
        choices = ["impatient", "frustrated", "anxious", "confused"]
        lo, hi = 0.52, 0.90
    else:
        choices = ["uncertain", "confused", "frustrated", "calm", "anxious"]
        lo, hi = 0.38, 0.80
    return rng.choice(choices), round(rng.uniform(lo, hi), 2)


def _next_emotion(rng: random.Random, current: str, intensity: float, phase: str, conflict: bool, profile: dict) -> tuple[str, float]:
    patience = profile["patience"]
    if phase == "impact":
        pool = ["anxious", "impatient", "frustrated", "uncertain", "disappointed"]
        delta = rng.uniform(-0.02, 0.12)
    elif phase == "repeat":
        pool = ["frustrated", "impatient", "skeptical", "angry", "disappointed"]
        delta = rng.uniform(0.03, 0.16) if patience < 70 else rng.uniform(-0.02, 0.08)
    elif phase == "apparent":
        pool = ["hopeful", "uncertain", "skeptical", "confused"] if conflict else ["hopeful", "relieved", "curious"]
        delta = rng.uniform(-0.16, 0.04)
    elif phase == "root_cause":
        pool = ["curious", "hopeful", "relieved", "skeptical", "appreciative"]
        delta = rng.uniform(-0.22, -0.04)
    else:
        pool = ["relieved", "appreciative", "satisfied", "calm", "hopeful", "skeptical"]
        delta = rng.uniform(-0.25, -0.06)
    new_intensity = min(0.98, max(0.20, intensity + delta))
    return rng.choice(pool), round(new_intensity, 2)


def _cue(rng: random.Random, emotion: str, intensity: float) -> str:
    cues = {
        "calm": ["steady pace", "even tone"],
        "neutral": ["matter-of-fact delivery", "steady pacing"],
        "curious": ["questioning inflection", "engaged tone"],
        "hopeful": ["lighter tone", "tentative optimism"],
        "appreciative": ["warm tone", "relaxed pacing"],
        "satisfied": ["confident tone", "relaxed delivery"],
        "relieved": ["audible relief", "slower exhale"],
        "uncertain": ["hesitation", "rising intonation"],
        "confused": ["frequent pauses", "questioning tone"],
        "anxious": ["faster speech", "tense delivery"],
        "disappointed": ["flat tone", "lower energy"],
        "frustrated": ["clipped phrasing", "audible tension"],
        "angry": ["raised intensity", "sharp emphasis"],
        "impatient": ["fast pace", "short responses"],
        "skeptical": ["guarded tone", "challenging questions"],
        "distressed": ["strained voice", "uneven pacing"],
        "embarrassed": ["quiet delivery", "self-conscious hesitation"],
    }
    base = rng.choice(cues.get(emotion, ["natural speech"]))
    return f"{base}; intensity {intensity:.0%}"


def _join_emotion_prefix(prefix: str, text: str) -> str:
    """Join a complete mood cue without damaging sentence capitalization."""
    if not prefix or not text:
        return text
    return prefix + text


def _style_text(rng: random.Random, text: str, emotion: str, intensity: float, profile: dict) -> str:
    if rng.random() > 0.78:
        return text
    # Each cue is a complete sentence, so the original scenario sentence keeps its
    # intended capitalization. Positive cues are deliberately mild and are only used
    # when the simulated emotion itself is positive.
    prefix_options = {
        "angry": ["I need a straight answer here. ", "This is really frustrating. "],
        "frustrated": ["This is getting frustrating. ", "I'm frustrated that this is still happening. "],
        "impatient": ["I need a quick answer here. ", "I'd really like to get to the actual fix. "],
        "anxious": ["I'm a little worried about this. ", "I really need some clarity here. "],
        "confused": ["I'm confused about what's happening. ", "I don't understand why these two things don't match. "],
        "skeptical": ["I need you to be certain here. ", "I don't want to rely on the screen if the system says something different. "],
        "disappointed": ["I'm disappointed this is still happening. ", "I was hoping this would be straightforward. "],
        "distressed": ["I really need help with this now. ", "This is becoming pretty stressful. "],
        "hopeful": ["That sounds more promising. ", "Hopefully we're close to getting this resolved. "],
        "relieved": ["Okay, that's helpful. ", "That helps. "],
        "appreciative": ["Thanks for checking. ", "I appreciate the help. "],
        "curious": ["I just want to make sure I understand. ", "Can you clarify one thing for me? "],
    }
    prefixes = prefix_options.get(emotion)
    styled = _join_emotion_prefix(rng.choice(prefixes), text) if prefixes and text else text
    style = profile.get("communication_style")
    if style == "concise" and len(styled) > 145:
        styled = styled.split(". ")[0].rstrip(".") + "."
    return styled

def _make_turn(
    rng: random.Random, text: str, emotion: str, intensity: float, profile: dict, *, decorate: bool = True
) -> CustomerTurn:
    return CustomerTurn(
        text=_style_text(rng, text, emotion, intensity, profile) if decorate else text,
        emotion=emotion,
        emotion_intensity=intensity,
        nonverbal_cue=_cue(rng, emotion, intensity),
    )


def generate_scenario(controls: ScenarioControls) -> GeneratedScenario:
    seed = controls.seed if controls.seed is not None else random.SystemRandom().randint(1, 2_000_000_000)
    rng = random.Random(seed)
    blueprints = _blueprints()
    domain = controls.domain if controls.domain in blueprints else rng.choice(list(blueprints))
    b = blueprints[domain]
    profile = _profile(rng, domain)

    # Conflict is deliberately always randomized; there is no user-facing conflict switch.
    include_conflict = rng.random() < 0.58
    status_type, status_value, status_evidence = b["status_event"]

    emotion, intensity = _initial_emotion(rng, profile)
    steps: list[ScenarioStep] = []

    steps.append(ScenarioStep(
        label="Opening issue",
        customer_turn=_make_turn(rng, b["opening"], emotion, intensity, profile),
        backend_events=[
            _event("profile", "customer_domain", domain, evidence=f"customer service domain={domain}", relevance=0.99, confidence=0.99),
            _event("relationship", "relationship_state", profile["relationship"], evidence=f"CRM relationship={profile['relationship']}", relevance=0.48, confidence=0.95),
        ],
    ))

    emotion, intensity = _next_emotion(rng, emotion, intensity, "impact", include_conflict, profile)
    steps.append(ScenarioStep(
        label="Customer explains impact",
        customer_turn=_make_turn(rng, b["impact"], emotion, intensity, profile),
        backend_events=[
            _event(status_type, "authoritative_status", status_value, evidence=status_evidence,
                   relevance=0.99, confidence=0.995, conflict_importance=0.72),
        ],
    ))

    # A third customer turn makes the conversation feel like a real troubleshooting exchange,
    # while emotion and intensity still vary with patience/relationship context.
    emotion, intensity = _next_emotion(rng, emotion, intensity, "repeat", include_conflict, profile)
    steps.append(ScenarioStep(
        label="Customer adds prior context",
        customer_turn=_make_turn(rng, b["followup"], emotion, intensity, profile),
    ))

    if include_conflict:
        emotion, intensity = _next_emotion(rng, emotion, intensity, "apparent", True, profile)
        steps.append(ScenarioStep(
            label="Customer sees apparently conflicting evidence",
            customer_turn=_make_turn(rng, b["apparent"], emotion, intensity, profile),
            image_observations=[ImageObservation(
                description=b["visual"],
                concept_name="customer_visible_status",
                concept_value=b["visible_value"],
                confidence=round(rng.uniform(0.78, 0.93), 2),
                relevance=round(rng.uniform(0.60, 0.82), 2),
                conflict_importance=0.72,
            )],
        ))

    emotion, intensity = _next_emotion(rng, emotion, intensity, "root_cause", include_conflict, profile)
    root_question = rng.choice([
        "What did you find, and can you fix the actual cause from your side?",
        "So what is causing this, and can you take care of it from your side?",
        "What is the underlying issue? If you can fix it on your side, please do.",
    ])
    steps.append(ScenarioStep(
        label="Diagnostic result becomes available",
        customer_turn=_make_turn(rng, root_question, emotion, intensity, profile),
        backend_events=[
            _event("diagnostic", "root_cause", b["root_cause"], evidence=b["root_evidence"], relevance=1.0, confidence=0.99),
        ],
    ))

    # Once the diagnosis is known, the next customer turn authorizes the fix and the
    # simulated company system completes it on that same turn.  This represents the
    # few minutes a real support agent would spend doing the work without forcing the
    # customer through several artificial "I'm still checking" turns.
    emotion, intensity = _next_emotion(rng, emotion, intensity, "closing", include_conflict, profile)
    remediation_request = rng.choice([
        "Okay, that makes sense. Please go ahead and fix it.",
        "Understood. Please make that correction so we can get this resolved.",
        "Thanks for explaining it. Yes, please take care of the fix on your side.",
    ])
    steps.append(ScenarioStep(
        label="Issue resolved after remediation",
        customer_turn=_make_turn(rng, remediation_request, emotion, intensity, profile, decorate=False),
        backend_events=[
            _event(
                "resolution", "authoritative_status", "resolved",
                evidence="support remediation completed and the system-of-record confirms resolution",
                relevance=1.0, confidence=0.995, conflict_importance=0.0,
            ),
        ],
    ))

    # After the agent confirms the completed fix, the customer closes naturally.
    emotion = rng.choice(["satisfied", "relieved", "appreciative", "calm"])
    intensity = round(rng.uniform(0.35, 0.68), 2)
    steps.append(ScenarioStep(
        label="No other concerns",
        customer_turn=_make_turn(
            rng,
            rng.choice([
                "No, that's everything. Thanks for getting it sorted out.",
                "That's all I needed. Thank you for fixing it.",
                "No other questions. I appreciate the help.",
            ]),
            emotion, intensity, profile, decorate=False,
        ),
    ))

    critical = ["authoritative_status", "root_cause"]
    if include_conflict:
        critical.append("customer_visible_status")

    return GeneratedScenario(
        scenario_id=f"scn_{uuid.uuid4().hex[:10]}",
        domain=domain,
        title=b["title"],
        problem_summary=b["opening"],
        customer_profile=profile,
        hidden_ground_truth={
            "authoritative_status": status_value,
            "root_cause": b["root_cause"],
            "customer_visible_evidence": b["visible_value"],
            "conflict_injected": include_conflict,
        },
        expected_conflict=include_conflict,
        critical_concepts=critical,
        steps=steps,
        seed=seed,
    )


def generate_manual_context(domain: str, seed: int | None = None) -> tuple[dict, list[BackendEvent]]:
    rng = random.Random(seed if seed is not None else random.SystemRandom().randint(1, 2_000_000_000))
    blueprints = _blueprints()
    if domain not in blueprints:
        domain = rng.choice(list(blueprints))
    b = blueprints[domain]
    profile = _profile(rng, domain)
    status_type, status_value, status_evidence = b["status_event"]
    # Manual mode now uses the same pacing idea as Scenario Lab: the company record
    # does not reveal the root cause on the very first customer message.  The hidden
    # case plan is carried in SessionState.manual_context and released as the
    # conversation progresses, which prevents the old three-turn diagnose/fix/bye loop.
    root_event = _event(
        "diagnostic", "root_cause", b["root_cause"], evidence=b["root_evidence"],
        relevance=1.0, confidence=0.99,
    )
    profile["_manual_case"] = {
        "opening": b["opening"],
        "impact": b["impact"],
        "followup": b["followup"],
        "apparent": b["apparent"],
        "final": b["final"],
        "root_cause": b["root_cause"],
        "root_evidence": b["root_evidence"],
        "root_cause_event": root_event.model_dump(),
        "root_questions": [
            "What did you find, and can you fix the actual cause from your side?",
            "So what is causing this, and can you take care of it from your side?",
            "What is the underlying issue? If you can fix it on your side, please do.",
        ],
        "fix_requests": [
            "Okay, that makes sense. Please go ahead and fix it.",
            "Understood. Please make that correction so we can get this resolved.",
            "Thanks for explaining it. Yes, please take care of the fix on your side.",
        ],
        "closings": [
            "No, that's everything. Thanks for getting it sorted out.",
            "That's all I needed. Thank you for fixing it.",
            "No other questions. I appreciate the help.",
        ],
    }
    events = [
        _event("profile", "customer_domain", domain, evidence=f"customer service domain={domain}", relevance=0.99, confidence=0.99),
        _event("relationship", "relationship_state", profile["relationship"], evidence=f"CRM relationship={profile['relationship']}", relevance=0.48, confidence=0.95),
        _event(status_type, "authoritative_status", status_value, evidence=status_evidence,
               relevance=0.99, confidence=0.995, conflict_importance=0.72),
    ]
    return profile, events
