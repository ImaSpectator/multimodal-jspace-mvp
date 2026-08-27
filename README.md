# JSpace Live — v1.3.2

## Updates

- Top-right controls remain a **borderless website-style toolbar** in normal page flow, avoiding the stacking/overlap issue caused by fixed positioning.
- Manual mode now has a deterministic **natural closing path**: once the issue is effectively resolved, conflicts are gone, and the customer is satisfied, the suggested customer move becomes a final thanks/goodbye. The agent replies once with a closing remark and the session ends immediately.
- Suggested customer moves now avoid reusing customer text already present in the transcript, reducing prompt cycling in longer Manual conversations.
- Rebuilt PDF chat rendering again as **full-width stacked transcript cards**. Customer and support messages no longer use staggered left/right bubbles, eliminating the crowded/overlapping appearance in long conversations and across PDF viewers.
- Post-session **Analyze conversation** and **Save conversation as PDF** remain available after the session ends.
- Moved the top-right toolbar higher and made the controls visually lighter: Help, link, reset, and settings are compact **borderless header icons** with only a subtle hover state.
- Fixed Manual suggested-prompt progression again. Each agent turn gets a new suggestion-button identity, and repeated states now rotate through multiple natural customer moves instead of getting stuck after the first couple of turns.
- Suggested customer moves continue to mix acknowledgements, instructions, corrections, authorization to proceed, and occasional targeted questions rather than creating endless question loops.
- Customer **starting Patience is scenario-dependent** rather than always 100. Relationship quality, recent support contacts, communication style, and urgency of the service domain influence the starting level.
- Patience still stays stable when support is moving forward and drops when the exchange stalls, repeats work, falls back, or leaves conflict unresolved. In Manual mode, if Patience falls below 0, the customer automatically ends the session.
- Added **Analyze conversation** after a conversation ends. DeepSeek produces a concise outcome/quality/JSpace analysis; a local analysis is used if the provider is unavailable.
- Added **Save conversation as PDF** after a conversation ends. The report includes the transcript, final Patience/Trust/Satisfaction, session metadata, provider labels, and the generated analysis when available. English and Simplified Chinese PDF output are both supported.
- Existing v1.1 behavior remains: dynamic Trust, localized Chinese JSpace concepts, settings persistence across tabs, mode-specific Manual inputs, Hy-ASR audio, YT-VITA video, DeepSeek text/images, bilingual scenarios, hidden scenario length, Settings-gated Researcher View, and clipboard-only link sharing.
- The global **English / 中文** language switch remains available across Scenario Lab, Manual mode, conversation analysis, and PDF export.
- Active model IDs: `deepseek/deepseek-v4-flash-vision-exp` for text/images, `hy-asr-3.0-preview` for standalone audio transcription, and `youtu-vita` for video understanding.

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
