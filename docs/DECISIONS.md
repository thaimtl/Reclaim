# Decisions, October 24

The team settled the conflicts between the EMG plan and the earlier README, CLAUDE.md and HARDWARE.md. This page records what was decided and what replaced what. The plan that follows from it is [PLAN.md](PLAN.md).

| # | Topic | Decision | Replaces |
| --- | --- | --- | --- |
| 1 | Who it's for and who demos | Laryngectomy patients are the audience. Austin, who is healthy, is the demo speaker; no patients take part in the demo. Performance and proof of concept come first, to impress the judges | "Healthy speaker only" (EMG plan) vs "patients" (README): both now hold, for different roles |
| 2 | Electrode sites | Gaddy sites 1, 2, 3, 5, 6, 8. No throat pad | Site 4 on the throat (EMG plan) |
| 3 | Reference pads | Each board has its own REF pad behind the ears, as in HARDWARE.md | One shared reference (EMG plan) |
| 4 | Modes | One mode: camera + EMG + Claude together, for free speech in real time. No separate camera mode or EMG mode. No closed-mouth speech, and no EMG training for it | Free speech camera-only plus a separate quick-phrases EMG mode (README); lips-sealed EMG mode (EMG plan) |
| 5 | Mouth | Open mouth, silent, camera always on | Lips sealed (EMG plan); camera off for quick phrases (README) |
| 6 | Fixing mistakes | Claude through the Claude API, for the Anthropic track. Claude follows the whole conversation, including the partner's words from Cartesia Ink. No local LLM | Local Qwen-class model (EMG plan) |
| 7 | When the voice speaks | Cartesia speaks in real time while Austin mouths, without waiting for the end of the sentence | Speaker confirms each sentence before playback (README, sponsor notes) |
| 8 | What gets trained | Only the EMG model (SVR-1), plus the decoder's weights fitted on Austin's recordings. Auto-AVSR is used as is; Claude is prompted | "No training" for free speech (README) |
| 9 | Phrase picking | Quick phrases are dropped, so the 10-phrase classifier and the Jev context check go with them. Claude covers context | Classifier top 3 + Jev (README) |
| 10 | Recording | About 75 minutes of open-mouth silent mouthing of varied sentences, with camera and EMG on one clock. No 50-repetition phrases | ~2.5 h with closed-mouth sets (EMG plan); ~30 min of phrases (README) |
| 11 | Compute | Everything runs on Austin's laptop (RTX 5060, 115 W): camera, EMG, decoder, Claude calls, Cartesia, dashboard | Thai's Mac reading the pads, linked to Austin's laptop over a hotspot (HARDWARE.md) |
| 12 | Streaming lip reading | Evan builds it, following [EVAN_WORKFLOW.md](EVAN_WORKFLOW.md), which was written with the whole system in mind | Evan to design it alone |
| 13 | The demo's headline | Real-time free speech: live captions of both speakers, and Austin's cloned voice speaking while he mouths, with his facial emotion | Free speech plus quick phrases with the camera off (README); fused speed + closed-lips switch (EMG plan) |

## New decisions

- **Emotion:** the camera reads Austin's facial expression (Evan), and Cartesia applies it to the voice through `generation_config` (Frank).
- **The partner's speech:** a microphone records the partner, Cartesia Ink transcribes it into the captions, and Claude reads it as context.
- **Speaker control:** "confirm before playback" is gone (decision 7). Instead, a deliberate jaw clench, read by the EMG jaw sites, cancels the voice mid-sentence, and Claude answers "please repeat" when it cannot tell what was said. This keeps the responsibility story for the Cartesia judges.
- **SVR-1 changes:**
    - dropped: the fixed-phrase head with its "none" class, closed-lips transfer and the local LLM;
    - added: a speaking-or-not output, partial-sentence training, pace and effort for the voice, and the jaw-clench detector.

  Details: [ARCHITECTURE.md](ARCHITECTURE.md#the-emg-reader-svr-1-revised).
- **Cartesia:** Sonic 3.6, with speed, volume and emotion through `generation_config`, per Cartesia's hackathon rules in `sponsors/cartesia.md`.

## Still open

- **Electrical safety with one laptop:** the pads now connect to the laptop that charges. Either a galvanically isolated USB link between the Arduino and the laptop, or the laptop runs on battery whenever pads are on skin (the GPU then slows down). See [HARDWARE.md](../HARDWARE.md#safety). Thai decides, before pads go on skin.
- **Owners** of the Claude predict/repair service (proposed: Frank) and the captions dashboard (proposed: Thai).
- **Claude model:** `claude-opus-5-5` at low effort is the default; `claude-haiku-5-5` if measured repair latency is too high.
