# Multimodal JSpace — Automated Customer Service MVP v0.2

A runnable research MVP for **capacity-limited and conflict-aware shared representations** in customer service.

The project now contains two modes:

1. **Automated Scenario Lab** — generate a complete synthetic customer-service case, simulate customer/backend/image signals, run the JSpace pipeline, and score the final decision automatically.
2. **Manual JSpace Sandbox** — type your own customer turns and inject structured backend evidence manually.

The MVP is intentionally API-key-free. It uses a deterministic synthetic scenario generator and a rule-based response layer so the JSpace research logic can be tested independently. OpenAI Realtime / multimodal perception can be plugged in later.

## Automated domains

- Account access / lockout
- Delivery / missing shipment
- Insurance claim support
- Internet / router outage
- Payment failure
- Return / refund
- Subscription cancellation
- Travel / booking change

The generator varies customer persona, audio cue, distracting backend context, difficulty, and whether a conflict is injected.

## What one automated run does

```text
Scenario Generator
      ↓
Hidden ground truth
      ↓
Customer simulator + backend simulator + image observations
      ↓
Concept extraction
      ↓
Concept normalization / update
      ↓
Conflict detection
      ↓
Capacity ranking (Top-K)
      ↓
Conflict-preserving JSpace
      ↓
Recommended action + customer-facing response
      ↓
Automatic evaluator
```

The evaluator checks:

- Whether the expected next action was chosen
- Whether an expected conflict was detected
- Whether critical evidence survived the Top-K capacity filter
- A weighted 0–100 score

## Example

A travel scenario can generate:

- Customer says the flight change appears successful
- App screenshot shows the new itinerary
- Reservation backend says the ticket reissue actually failed
- Customer sounds uncertain

The final JSpace can preserve:

```text
authoritative_status -> unresolved
customer_belief_status -> resolved
customer_domain -> travel
root_cause -> ticket reissue failed after itinerary change
customer_sentiment -> uncertain
```

and explicitly detect:

```text
Customer believes the issue is resolved,
but the authoritative system says unresolved.
```

## Run locally

Recommended: Python 3.11 or 3.12.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Terminal 1 — API

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

Check:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

### Terminal 2 — Streamlit UI

```bash
python -m streamlit run frontend/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Automated API example

```bash
curl -X POST http://localhost:8000/scenarios/autorun \
  -H "Content-Type: application/json" \
  -d '{
    "controls": {
      "domain": "travel",
      "difficulty": "hard",
      "include_conflict": true,
      "seed": 77
    },
    "capacity_k": 5,
    "preserve_conflicts": true
  }'
```

Useful routes:

- `GET /health`
- `GET /scenarios/domains`
- `POST /scenarios/generate`
- `POST /scenarios/autorun`
- `POST /scenarios/run`
- `POST /sessions`
- `POST /sessions/{id}/turn`
- `POST /sessions/{id}/backend-event`
- `POST /sessions/{id}/image-observation`

## Tests

```bash
PYTHONPATH=. pytest -q
```

Current included suite covers every automated domain with conflict and non-conflict variants.

During development, the simulator was additionally stress-tested over 400 generated runs across all 8 domains and multiple seeds with no failures in the current deterministic harness.

## Important research limitation

The current generator, concept extractor, customer response, and evaluator are mostly deterministic/rule-based. That is useful for verifying the JSpace mechanics, but the current 100/100 automated score **is not evidence that JSpace beats a frontier multimodal LLM**.

The next research-valid version should plug in a real multimodal/voice model and compare the exact same held-out scenarios across:

- Direct multimodal model
- Naive fusion
- Capacity-only JSpace
- Conflict-only JSpace
- Full JSpace

A fixed human-reviewed benchmark should be used for the final reported comparison so the same generator is not both creating and grading all cases.

## Next integrations

The code is structured so the following can be added without replacing the JSpace engine:

- OpenAI multimodal concept extraction from real screenshots/images
- OpenAI Realtime or another low-latency speech model
- LiveKit/WebRTC for live microphone calls and barge-in
- Real CRM/backend tool adapters
- LLM-generated scenario variants
- Human-reviewed benchmark and experiment dashboard
