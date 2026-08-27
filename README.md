# JSpace Live — v1.0

## Updates

- Fixed the Manual **Use prompt** crash. Suggested text is now queued before Streamlit creates the chat widget, so clicking **Use prompt** safely fills the textbox on the next rerun and can be edited before Enter/Send.
- Manual Multimodal AI keeps five distinct input modes: **Text Messages**, **Image Upload**, **Audio Upload**, **Video Upload**, and **Multimodal Mix**.
- **Text Messages** accepts text only. **Image Upload** accepts images only. **Audio Upload** accepts audio only. **Video Upload** accepts video only. **Multimodal Mix** accepts text, images, audio, and video.
- Suggested prompts appear only in **Text Messages** and **Multimodal Mix** and evolve using the latest agent reply, conflict state, session phase, and conversation progress.
- Evidence controls are now visibly open by default and use mode-specific headings such as **Image evidence**, **Audio evidence**, **Video evidence**, and **Multimodal evidence**.
- The global **English / 中文** toggle keeps generated customer turns, DeepSeek agent replies, local fallback replies, and suggested prompts in the selected language. Chinese mode has a deterministic Simplified-Chinese scenario fallback if AI scenario rewriting fails.
- The main JSpace display prioritizes task-specific concepts. Routine resolution/status concepts are kept in a separate lower **Resolution/status context** lane so they do not dominate the primary workspace.
- Scenario conversation length/progress counters are hidden.
- **Researcher View** is hidden by default and only appears after it is explicitly enabled in **Settings**.
- The top-right toolbar uses compact controls with stricter horizontal/vertical centering.
- **Share** is now a clipboard-only action that copies the current page link. There is no email/share workflow.
- Text/images use `deepseek/deepseek-v4-flash-vision-exp`; standalone audio uses `hy-asr-3.0-preview`; video uses `youtu-vita`.

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
