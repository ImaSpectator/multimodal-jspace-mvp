# JSpace Live — v1.4.1

## Updates

- PDF export is redesigned as an archive-friendly conversation report with session details, readable metric cards, full-width message records, page numbers, and analysis on a separate page.
- Manual suggested customer messages now sound like customer speech rather than diagnostic instructions; acknowledgements and context statements are allowed instead of forcing a question every turn.
- Manual support replies treat the simulated case state as the completed result of a backend check, so the agent says whether the issue is confirmed/unconfirmed instead of repeatedly promising to verify it later.
- A natural fourth-turn customer authorization can complete the simulated remediation after diagnosis, avoiding the old verify-again loop.
- Settings gear optical offset moved 2 px left (6 px -> 4 px).
- Manual Multimodal AI now follows a **Scenario-Lab-like conversation arc** instead of preloading the diagnosis and collapsing into the same three turns. The simulated backend starts unresolved, reveals the root cause after a short discovery exchange, then waits for a later explicit customer authorization before marking the remediation resolved.
- Suggested customer prompts are now **stage-aware and session-varied**. They progress through opening, impact, prior context, diagnosis discussion, remediation authorization, confirmation, and a natural goodbye while still reacting to the latest agent reply. Different sessions rotate through different wording instead of always showing the same few prompts.
- Manual agent instructions now enforce more realistic pacing: early turns report the current simulated backend result directly, diagnosis turns explain the connection between cause and symptom, and the agent cannot claim a fix completed until the authoritative backend status is actually resolved.
- The resolved customer closes naturally, the agent gives one final thank-you/goodbye, and the session ends immediately; analysis and PDF export remain available afterward.
- Existing multimodal routing remains unchanged: `deepseek/deepseek-v4-flash-vision-exp` for text/images, `hy-asr-3.0-preview` for standalone audio transcription, and `youtu-vita` for video understanding.
- The global **English / 中文** switch remains available across Scenario Lab, Manual mode, analysis, and PDF export.

## Speed profiles

Speed and reply length are independent.

| Profile | DeepSeek attempt cap | Attempts | Conversation context | Scenario wording cap |
| --- | ---: | ---: | ---: | ---: |
| **Fast** | 12 seconds | Up to 2 | 4 recent messages | 12 seconds |
| **Balanced** | 20 seconds | Up to 2 | 6 recent messages | 20 seconds |

Both profiles keep DeepSeek thinking disabled and stream output as it arrives.

**Reply length:**

- **Concise** — up to 120 output tokens, targeting about 2 sentences.
- **Standard** — up to 180 output tokens, targeting about 3 sentences.

## How to run locally

From the project root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```
