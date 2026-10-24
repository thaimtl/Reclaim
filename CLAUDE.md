# Reclaim

Cal Hacks 13.0 project, October 23 to 25, 2026: silent speech for people who lost their voice box.
Read README.md and HARDWARE.md before answering.

Decisions already made, so don't re-litigate them without new evidence (details in docs/DECISIONS.md):
- One mode: free speech in real time, open mouth, silent. Camera + EMG + Claude together; no separate camera or EMG mode, no closed-mouth speech, no quick phrases, no Jev.
- The audience is laryngectomy patients; the demo speaker is Austin, who is healthy. Performance comes first.
- Camera: Auto-AVSR as is, made real time by re-reading the current segment every 100 ms (docs/EVAN_WORKFLOW.md). No recording or fine-tuning for it.
- EMG: our model SVR-1, the only thing we train, on 6 BioAmp EXG Pill channels at Gaddy sites 1, 2, 3, 5, 6, 8 into an Arduino Uno. It proposes words, settles lip look-alikes inside the joint search, detects speech start, word ends and sentence end, carries captions when the face is hidden, gives the voice its pace and effort, and tells Claude which words are uncertain.
- Claude follows the whole conversation (Anthropic track), through the Claude API (`claude-opus-5-5`, low effort); no local LLM. It predicts likely words before Austin mouths them and repairs look-alikes. It only swaps look-alike words and never adds ideas.
- A word locks (goes to captions and voice) when one learned confidence score from camera, EMG and Claude crosses a threshold that moves with the conversation; Claude alone can never lock a word. Unclear words end in "Please repeat".
- The demo is a Twilio phone call: the judge speaks, Austin mouths, his cloned voice goes into the call. Nothing is trained on the pitch.
- Everything is logged at all times, with a replay tool.
- Cartesia Sonic 3.6 speaks in Austin's cloned voice while he mouths, without waiting for the sentence to end, with emotion from his face. Cartesia Ink transcribes the judge from the call into the captions and into Claude's context.
- No pad on the throat, no AD8232, no ESP32 in the rig, and each board has its own REF pad.
- Everything runs and is tested on Austin's laptop (RTX 5060). The Uno connects through a USB isolator, or the laptop stays on battery while pads are on skin.
- Parts talk over ZeroMQ on the laptop, with the message formats in docs/PLAN.md.

Never commit API keys or sponsor codes.
Diagrams are Mermaid in Markdown.
Read `sponsors/*.md` before any pitch, demo, or sponsor-facing work.
Every change goes through a PR. Pushing straight to main is blocked.
