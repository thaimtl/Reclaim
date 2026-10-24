# Decisions, October 24

The team settled the conflicts between the EMG plan and the earlier README, CLAUDE.md and HARDWARE.md, then refined the design in a second pass the same day. This page records what was decided and what each decision replaced. The plan that follows from it is [PLAN.md](PLAN.md).

## The conflicts

| # | Topic | Decision | Replaces |
| --- | --- | --- | --- |
| 1 | Who it's for and who demos | Laryngectomy patients are the audience. Austin, who is healthy, is the demo speaker; no patients take part. Performance and proof of concept come first, to impress the judges | "Healthy speaker only" (EMG plan) vs "patients" (README): both hold now, for different roles |
| 2 | Electrode sites | Gaddy sites 1, 2, 3, 5, 6, 8. No throat pad | Site 4 on the throat (EMG plan) |
| 3 | Reference pads | Each board has its own REF pad behind the ears, as in HARDWARE.md | One shared reference (EMG plan) |
| 4 | Modes | One mode: camera + EMG + Claude together, for free speech in real time. No separate camera or EMG mode, no closed-mouth speech, no EMG training for it | Free speech camera-only plus a quick-phrases EMG mode (README); lips-sealed EMG mode (EMG plan) |
| 5 | Mouth | Open mouth, silent, camera always on | Lips sealed (EMG plan); camera off for quick phrases (README) |
| 6 | Fixing mistakes | Claude through the Claude API (`claude-opus-5-5`, low effort), for the Anthropic track. Claude follows the whole conversation, both sides, and predicts likely words before Austin mouths them. No local LLM | Local Qwen-class model (EMG plan) |
| 7 | When the voice speaks | Cartesia speaks in real time while Austin mouths, without waiting for the end of the sentence | Speaker confirms each sentence before playback (README, sponsor notes) |
| 8 | What gets trained | Only the EMG model (SVR-1), plus the decoder's weights, gate and lock model fitted on Austin's recordings. Auto-AVSR is used as is; Claude is prompted | "No training" for free speech (README) |
| 9 | Phrase picking | Quick phrases are dropped, so the 10-phrase classifier and the Jev context check go with them | Classifier top 3 + Jev (README) |
| 10 | Recording | Three sessions of varied conversational sentences, mouth open, camera and EMG on one clock: sessions 1 and 2 (250 sentences each) train, session 3 (100 sentences, in demo conditions) is the test. Pads re-applied between sessions. Nothing from the pitch | ~2.5 h with closed-mouth sets (EMG plan); ~30 min of phrases (README) |
| 11 | Compute | Everything runs on Austin's laptop (RTX 5060, 115 W), and all testing happens there | Thai's Mac reading the pads, linked over a hotspot (HARDWARE.md) |
| 12 | Streaming lip reading | Evan builds it, following [EVAN_WORKFLOW.md](EVAN_WORKFLOW.md) | Evan to design it alone |
| 13 | The demo | Austin calls a judge through Twilio and mouths the whole conversation. The judge speaks normally; captions show both sides; Austin's cloned voice goes into the call in real time, with his facial emotion. The pitch is given separately, with real voices | Free speech plus quick phrases with the camera off (README); fused speed + closed-lips switch (EMG plan) |

## The second pass

- **Re-reading every 100 ms.** The camera and EMG models re-read the current sentence every 100 ms (a setting, lower if the laptop allows).
- **Locking is purely confidence-based.** Each word gets one score per pass from all ingredients:
    - camera support, EMG support and their agreement;
    - Claude's prediction, with Claude's share capped so it can never lock a word alone;
    - whether the word is finished;
    - how many passes it has survived;
    - signal quality.

  A small model learns the weights from sessions 1 and 2. A word locks the moment its score crosses the threshold, which can be on its first pass. The threshold moves with how predictable Claude finds the conversation. There is no separate fixed rule such as "the same word twice".
- **"Please repeat".** If a word is still not confident after the end-of-sentence pass, the screen says "Please repeat" and the debug panel shows which reader disagreed.
- **The error log.** Every signal and every decision is logged to disk at all times, with a replay tool to reproduce any session.
- **EMG extras, all in:**
    - it proposes words and settles look-alikes inside the joint search;
    - it detects speech start, word ends and sentence end first;
    - it keeps captions going when the face is hidden;
    - it drives the voice's speed and volume from pace and effort;
    - it tells Claude which words are uncertain.
- **Jaw-clench cancel is dropped.** The voice plays in real time, so there is no time to react and cancel; the speaker's safeguards are "Please repeat" and captions that show exactly what was said.
- **ZeroMQ.** Small separate programs on the laptop talk over ZeroMQ on local sockets, well under a millisecond per message. The logger is one more listener.
- **Owners:**
    - Evan: the camera reader;
    - Austin: EMG model, decoder, locking, logging;
    - Frank: Cartesia, Twilio, the captions dashboard;
    - Thai: the EMG rig and the Claude service.
- **Phone call delay is accepted.** The phone network's delay is not ours to remove; everything else is built for the lowest delay, and the dashboard shows the real latency.
- **Cartesia:** Sonic 3.6, with speed, volume and emotion through `generation_config`. The emotion names the camera produces are accepted.
- **Electrical safety:** a USB isolator between the Arduino and the laptop, or the laptop on battery while pads are on skin.
