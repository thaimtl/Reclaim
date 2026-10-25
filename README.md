# Reclaim

Silent speech for people who lost their voice box.
You mouth the words, and Reclaim says them out loud in your own voice, while you are still mouthing, even on a phone call.

Built at Cal Hacks 13.0, October 23 to 25, 2026.

## Who it's for

A laryngectomy removes the voice box, usually after throat cancer.
People can still move their lips, tongue, and jaw, so the words are there without the sound.
Most rely on an electrolarynx, a handheld buzzer that sounds robotic and needs a free hand.

Our demo speaker is Austin, who is healthy; no patients take part in the demo.

## How it works

One mode: open mouth, no sound, real time.
Three readers work together, because each sees what the others can't, and every 100 ms one joint search combines them and locks in each word as soon as they're confident enough.

```mermaid
flowchart LR
    W[Webcam] --> AV[Camera reader<br/>Auto-AVSR]
    P[4 EMG pads] --> E[EMG reader<br/>SVR-1]
    AV --> D[Joint search + language model]
    E --> D
    M[Phone call<br/>the other person] --> I[Cartesia Ink]
    I --> C[Claude<br/>whole conversation]
    C <--> D
    D --> CAP[Live captions<br/>both speakers]
    I --> CAP
    D --> V[Cartesia Sonic<br/>cloned voice + emotion<br/>into the call]
```

- The camera reads the lips and does most of the work. It also reads the facial expression, which sets the voice's emotion.
- The EMG model reads the face muscles, so it sees the tongue and jaw that no camera can. It settles words that look alike on the lips ("bat", "mat", "pat"), proposes its own guesses, and keeps going when the face is hard to see.
- Claude follows the whole conversation, both sides. It predicts likely words before you start mouthing, and it repairs look-alike mistakes ("bills" → "pills" after a question about medicine). It only swaps look-alike words and never adds ideas.
- Cartesia speaks each word in your cloned voice as soon as it locks, without waiting for the end of the sentence.
- If the system can't tell what you said, it shows "Please repeat" instead of guessing.

How the whole system works: [docs/REALTIME_PIPELINE.md](docs/REALTIME_PIPELINE.md). The EMG model, SVR-1: [docs/SVR1_SPEC.md](docs/SVR1_SPEC.md). Who builds what: [docs/TEAM_TASKS_AND_INTERFACES.md](docs/TEAM_TASKS_AND_INTERFACES.md). The rig: [HARDWARE.md](HARDWARE.md). Recording sessions: [RECORDING.md](RECORDING.md).

## What we demo

Two minutes, live, at our table.

1. Austin calls the judge, standing in front of us, through Twilio. The judge talks normally; their words appear in the captions.
2. Austin mouths his side of the conversation. His words appear as he mouths and play into the call in his cloned voice while he is still going.
3. His facial expression sets the voice's emotion.
4. The dashboard shows the real latency, which reader settled each word, and what Claude changed.

## Stack

| Part | Tool |
|---|---|
| Lip reading | [Auto-AVSR](https://github.com/mpc001/auto_avsr), as is, re-read every 100 ms for real time |
| Face, mouth crop, expression | MediaPipe FaceLandmarker |
| Muscle signals | BioAmp EXG Pill boards and an Arduino Uno, see [HARDWARE.md](HARDWARE.md) |
| EMG model | SVR-1, ours: PyTorch, pre-trained on Berkeley's silent speech data, fine-tuned on Austin |
| Context and repairs | Claude, through the Claude API |
| Voice clone and speech | Cartesia Sonic 3.6, Instant Voice Clone |
| The other person's speech to text | Cartesia Ink |
| Phone call | Twilio |
| Compute | One laptop, RTX 5060 |

## Code and data

All code lives in this repo; datasets, checkpoints, results and Python environments live in a working directory outside it.

| Folder | What it is |
| --- | --- |
| `svr/` | SVR-1, the EMG model: model, training, fine-tuning, preprocessing ([docs/SVR1_SPEC.md](docs/SVR1_SPEC.md)) |
| `fusion/` | The joint search over camera + EMG + Claude, and its language model: a conversational 4-gram with a cache of the call (`fusion/lm.py`, built by `fusion/build_lm.py`) |
| `bench/` | Benchmarks and their frozen test splits |
| `cluster/` | H100/H200 job scripts (PBS), smoke tests, runbook |
| `camera/` | Camera-only baselines (Auto-AVSR, BRAVEn, LipLearner) |
| `rig/` | EMG rig: Arduino firmware, serial reader, live hum filter, signal check, `emg.raw` publisher (`publish.py`) |
| `recorder/` | The recording app: camera + EMG + the session sentences ([recorder/README.md](recorder/README.md)) |
| `bus/` | The team bus: ZeroMQ wire format, the camera's topics (`__init__.py`) and every other endpoint and port (`team.py`) |
| `buslog/` | Logs every bus message to `$RECLAIM_WORKDIR/logs/` (`record.py`) and replays a session (`replay.py`) |
| `claude_service/` | The Claude service: predicts Austin's next words and repairs look-alikes (`brain.py`), on the bus (`service.py`) |
| `demo/dashboard/` | Captions dashboard prototype, sample data ([demo/dashboard/README.md](demo/dashboard/README.md)) |
| `call/` | The phone call: Twilio, Austin's cloned voice (Cartesia Sonic) and the judge's captions (Cartesia Ink) ([call/README.md](call/README.md)) |
| `figures/` | The SVR-1 architecture figure and the script that draws it |
| `tests/` | Smoke and unit tests |
| `docs/` | Plans and specs, Markdown with PDF copies |

The code finds data through the environment variable `RECLAIM_WORKDIR` (datasets in `silent_speech/` and `data/`, plus `checkpoints/`, `benchmark_results/`, `recordings/`, `third_party/`). Without it, data is expected next to the code, which is how the cluster copy works. Data folders are in `.gitignore`.

## Known limitations

- This is a prototype, not a medical device.
- Lip reading gets about 1 in 5 words wrong even on normal speaking video, and silent mouthing is harder; the EMG model and Claude are there to close that gap.
- The EMG model is trained for one person (Austin), from three recording sessions, see [RECORDING.md](RECORDING.md).
- The voice runs a little behind the lips, because a word is spoken only once it is confident enough to lock.

## Built on

- [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf), Gaddy and Klein, UC Berkeley, 2020, for the electrode sites and pre-training data.
- [Auto-AVSR](https://github.com/mpc001/auto_avsr), Ma et al., 2023, for lip reading.
- [Lipflow](https://github.com/amywork777/lipflow), MIT license, for live mouth cropping with MediaPipe.

## Team

Thai (EMG hardware and the Claude service), Austin (EMG model and fusion, and the speaker in the demo), Frank (Cartesia voice, the Twilio call and the captions dashboard), and Evan (real-time lip reading and facial emotion).
