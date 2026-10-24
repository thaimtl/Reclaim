# Lip reader (Evan)

The webcam goes through MediaPipe for the mouth crop and the blendshapes.
Every 100 ms, Auto-AVSR (as is) re-reads the current segment, and the result goes out as `cam.tick`: per-frame CTC top-32, blank, greedy text, activity, emotion and `timing_ms`.
At each segment end, a background beam search with the language model sends `cam.final` (5-best).
Everything travels on the team bus (`bus/`, ZeroMQ + msgpack on `tcp://127.0.0.1:5556`), whose contract is [bus/schema.md](../bus/schema.md).
The team interfaces are in `docs/TEAM_TASKS_AND_INTERFACES.md`, and the milestones in `docs/EVAN_REALTIME_LIP_READING.md`.
Data (`third_party/`, `recordings/`) is found through `RECLAIM_WORKDIR` (README, "Code and data").

## Setup (once, from the repo root)

On Windows, run these in Git Bash; the venv's Python is `venv-live/Scripts/python.exe` instead of `venv-live/bin/python`.

```bash
bash lipread/setup.sh                                   # venv-live, Lipflow pinned, weights, sample
venv-live/bin/python -m pytest tests/lipread tests/bus -q
venv-live/bin/python -m lipread.bench.laptop_check      # on Austin's laptop: driver, CUDA, webcam, power, encoder
```

## Run

```bash
venv-live/bin/python -m lipread.run --cam 0                       # webcam
venv-live/bin/python -m lipread.run --cam 0 --camera-settings     # Windows: lock exposure and focus in the dialog
venv-live/bin/python -m lipread.run --replay third_party/lipflow/samples/2016-03-12.mov --start 15 --end 37 --realtime
venv-live/bin/python -m lipread.run --replay s1_001               # a recording id (recordings/ labels.csv)
venv-live/bin/python -m bus.listen                                # second terminal: validates and prints every message
venv-live/bin/python -m bus.stub                                  # fake cam.tick / cam.final, no camera or GPU
venv-live/bin/python -m lipread.bench.tick_bench                  # tick p50/p95 at 2/4/8 s, fp32 vs fp16
```

Useful flags: `--size 1920x1080` (camera capture, default), `--process-size 1280x720` (frames shrunk before MediaPipe, default), `--half` (fp16 encoder, parity-checked), `--no-fe-cache`, `--no-emotion`, `--no-final`, `--segments stream`, `--tick-ms 200`, `--log ticks.jsonl`.

## Files

| File | What |
| --- | --- |
| `run.py` | the process: crop thread, then the tick loop (segments, re-read, publish), then the final pass |
| `source.py` | webcam (DirectShow on Windows, V4L2 on Linux) and replay sources |
| `crop.py`, `crop_view.py` | MediaPipe landmarks + blendshapes, Auto-AVSR-aligned 96×96 crops, debug view |
| `endpoint.py` | mouthing activity and segments (300 ms pre-roll, 0.5 s still ends, 8 s cap with 2 s carried over) |
| `model.py` | Auto-AVSR encoder + CTC top-k per read, front end / Conformer timing |
| `final.py` | beam 5 + LM in a background process, which publishes `cam.final` |
| `emotion.py` | emotion tag for the voice: the expression network when its model is there (on its own thread), else the blendshape rules |
| `expression.py` | HSEmotion expression network on the face crop + the head fitted on the speaker's sessions |
| `emotion_fit.py` | fits that head on `recordings/session_0*` and prints held-out scores (`--test session_02` scores a session it never saw) |
| `types.py` | `Frame` and `Crop`; the message types live in `bus/` |
| `bench/` | `laptop_check.py` (the laptop), `tick_bench.py` (tick speed) |

Recording uses the team's `recorder/`, and `--replay` reads its sessions.

Measured numbers and decisions: [MEASURED.md](MEASURED.md).
