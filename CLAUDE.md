# Reclaim

Cal Hacks 13.0 project, October 23 to 25, 2026: silent speech for people who lost their voice box.
Read README.md and HARDWARE.md before answering.

Decisions already made, so don't re-litigate them without new evidence (details in docs/DECISIONS.md):
- One mode: free speech in real time, open mouth, silent. Camera + EMG + Claude together; no separate camera or EMG mode, no closed-mouth speech, no quick phrases, no Jev.
- The audience is laryngectomy patients; the demo speaker is Austin, who is healthy. Performance comes first.
- Camera: Auto-AVSR as is, made real time by re-reading the current segment every 200 ms (docs/EVAN_WORKFLOW.md). No recording or fine-tuning for it.
- EMG: our model SVR-1, the only thing we train, on 6 BioAmp EXG Pill channels at Gaddy sites 1, 2, 3, 5, 6, 8 into an Arduino Uno. It proposes words, settles lip look-alikes, detects speech, and gives pace, effort and a jaw-clench cancel.
- Claude fixes the words using the whole conversation (Anthropic track), through the Claude API; no local LLM. It only swaps look-alike words, never adds ideas, and says "Please repeat" when unsure.
- Cartesia Sonic 3.6 speaks in Austin's cloned voice while he mouths, without waiting for the sentence to end, with emotion from his face. Cartesia Ink transcribes the partner into the captions and into Claude's context.
- No pad on the throat, no AD8232, no ESP32 in the rig, and each board has its own REF pad.
- Everything runs on Austin's laptop (RTX 5060). The Uno connects through a USB isolator, or the laptop stays on battery while pads are on skin.
- Parts talk over ZeroMQ on the laptop, with the message formats in docs/PLAN.md.

Never commit API keys or sponsor codes.
Diagrams are Mermaid in Markdown.
Read `sponsors/*.md` before any pitch, demo, or sponsor-facing work.
Every change goes through a PR. Pushing straight to main is blocked.
