# Resonant AI Public Pilot — Render Free

This is a **restricted visitor-only edition** extracted from RC F; the Windows research routes are intentionally not mounted.

## Publish without keeping a laptop running
1. Unzip this package, and upload the **contents of this folder** to a private GitHub repository. Do not upload credentials or participant recordings.
2. Open Render Dashboard → New → Blueprint → connect your GitHub repository → deploy the `render.yaml` Blueprint. Check that the plan says Free before approving.
3. Wait for the build, then open your `https://<your-service>.onrender.com/companion-experience` URL.
4. Test in mobile Safari/Chrome: logo → companion → language → question → Petra video → EEG → image upload. If it fails, inspect Render → Logs.

## Important limits
- This is a public **pilot**, not production-approved. Only the dedicated interactive-demo APIs are included, but public inputs should be subjected to further security review and real load testing before a conference.
- The Render Free instance may sleep while idle. The 58 MB Petra video and WebGL assets may be slow on mobile. Consider CDN delivery and compression before a large event.
- The visitor application needs a separately configured LLM service for open-ended answers and a neural TTS service for natural speech. Without these, it uses limited built-in answers and available browser voices. Never put provider secrets in JS or git; set Render Environment variables (`RESONANT_DEMO_LLM_URL`, `RESONANT_DEMO_LLM_MODEL`, `RESONANT_DEMO_LLM_KEY`, `RESONANT_DEMO_TTS_URL`, `RESONANT_DEMO_TTS_MODEL`, `RESONANT_DEMO_TTS_KEY`).
- A local Ollama model is intentionally disabled in this public edition. The synthetic EEG in the Petra tour is not participant data.
- Sessions are held in process memory, are lost on restart, and are not shared across instances. The project uses one worker only.
- Rate limiting is simple per-connection-IP, **not** comprehensive abuse protection; no user authentication is included.
- There is no guarantee Render Free has enough CPU/RAM for the full analysis engine; validate on the deployed instance before public distribution.

## Local sanity checks
`pip install -r requirements-render.txt`
`uvicorn app.public_main:app --host 127.0.0.1 --port 8765`
Open `http://127.0.0.1:8765/companion-experience`.
