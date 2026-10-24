# Live camera reader (Evan)

The webcam goes through MediaPipe for the mouth crop and the blendshapes. Every 100 ms, Auto-AVSR (as is) re-reads
the current segment. The result goes out as `cam.tick` (per-frame CTC top-32, blank, greedy text, activity, emotion,
`timing_ms`). At each segment end a background beam with the LM sends `cam.final` (5-best). Everything travels over
ZeroMQ + msgpack on `ipc:///tmp/reclaim/cam.tick`. The contract is in [bus_schema.md](bus_schema.md), the team
interfaces in `docs/TEAM_TASKS_AND_INTERFACES.md`, and the milestones in `docs/EVAN_REALTIME_LIP_READING.md`.
Backup recorder only: the team's recording tool is `recorder/`. Data (`third_party/`, `recordings/`) is found through
`RECLAIM_WORKDIR` (README, "Code and data").

## Setup (once, from the repo root)

```bash
bash live/setup.sh      # venv-live, Lipflow pinned, Auto-AVSR + LM + face landmarker weights, emotion model, sample
venv-live/bin/python -m pytest tests/live -q
bash live/bench/laptop_check.sh          # on Austin's laptop: driver, CUDA, webcam, encoder timings
```

## Run

```bash
venv-live/bin/python -m live.run --cam 0                       # webcam
venv-live/bin/python -m live.run --replay third_party/lipflow/samples/2016-03-12.mov --start 15 --end 37 --realtime
venv-live/bin/python -m live.run --replay s1_001                # a recording id (recordings/ manifest or labels.csv)
venv-live/bin/python -m live.bus_listen                         # second terminal: validates and prints every message
venv-live/bin/python -m live.stub_bus                           # fake cam.tick / cam.final, no camera or GPU
venv-live/bin/python -m live.bench.tick_bench                   # E3: tick p50/p95 at 2/4/8 s, fp32 vs fp16
```

Useful flags: `--size 1920x1080` (camera capture, default), `--process-size 1280x720` (frames shrunk before
MediaPipe, default), `--half` (fp16 encoder, parity-checked), `--no-fe-cache`, `--no-emotion`, `--no-final`,
`--segments stream`, `--tick-ms 200`, `--log ticks.jsonl`.

## Files

| File | What |
| --- | --- |
| `run.py` | the process: crop thread, then the tick loop (segments, re-read, publish), then the E6 final pass |
| `model.py` | Auto-AVSR encoder + CTC top-k per read, front end / Conformer timing |
| `crop.py`, `crop_view.py` | MediaPipe landmarks + blendshapes, Auto-AVSR-aligned 96×96 crops, debug view |
| `endpoint.py` | mouthing activity and segments (300 ms pre-roll, 0.5 s still ends, 8 s cap with 2 s carried over) |
| `final.py` | beam 5 + LM in a background process, which publishes `cam.final` |
| `emotion.py`, `emotion_net.py` | E5 emotion tag (blendshape rules + EmotiEffLib classifier; Frank) |
| `bus.py`, `bus_schema.md`, `bus_listen.py`, `stub_bus.py` | team bus contract (v2), listener, stub |
| `source.py`, `record.py` | webcam / replay sources; a backup recorder (the team recording tool is the main one) |

Measured numbers and decisions: [MEASURED.md](MEASURED.md).
