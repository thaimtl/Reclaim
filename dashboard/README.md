# Live dashboard

What the judges watch during the call, built from Thai's Figma "Reclaim — Demo Dashboard v2", section v5 (frames:
Reading, Claude repair, Please repeat, Face hidden, Muscle sensors off, End of call). The left side has the
conversation, Austin's words and who settled each one, and the latency. The right side has Austin's mirrored
camera with the four pads, the emotion and what the call heard.

```mermaid
flowchart LR
    B(("team bus<br/>5555 to 5560")) -- "every topic" --> S["dashboard/server.py<br/>CallState (state.py)"]
    S -- "snapshot JSON, 15 Hz" --> P["browser<br/>static/index.html"]
    S -- "each cam.preview frame + its face points, as it arrives" --> P
```

## Run

```bash
venv-call/bin/python -m dashboard.server --number "(628) 888-6598"     # then open http://127.0.0.1:8790, F11
```

It only listens, so it can start or stop at any time without touching the call.

**Pads on his face.** The pad dots follow his face like a camera filter: each video frame reaches the page with the
face points of that same frame and is drawn together with the pads, so they never trail the picture. A pad sits at
a face landmark plus offsets along his face's own axes, so it scales and turns with his head. Their labels sit in
columns beside his face, joined by lines. Pads go on in a slightly different place each session, so once the camera
shows his face, press **C** and click the four pads in the order shown (A0 under the chin, A1, A2, A3); it is saved in
that browser (Shift+C goes back to the defaults). No smoothing is added: MediaPipe's video mode already steadies its
points (0.44 px of jitter on the panel, measured), and a One Euro filter only made the pads lag up to 6 px. To rehearse without Austin, run
the scripted call with every part stopped. On a shared laptop, use private ports so no one's logger records it:

```bash
export RECLAIM_EMG_PUB=tcp://127.0.0.1:6555 RECLAIM_CAM_PUB=tcp://127.0.0.1:6556 RECLAIM_DEC_PUB=tcp://127.0.0.1:6557 \
       RECLAIM_CLAUDE_PUB=tcp://127.0.0.1:6558 RECLAIM_CALL_PUB=tcp://127.0.0.1:6559 RECLAIM_SVR_PUB=tcp://127.0.0.1:6560
venv-call/bin/python -m dashboard.server &
venv-live/bin/python -m dashboard.demo_feed --video some_clip.mp4     # --video needs OpenCV (venv-live)
```

A logged call replays onto it with `python -m buslog.replay SESSION`.

## Where each number comes from

Everything on the screen is measured on the call, from the bus. The rules:

- **Nothing is made up.** A part that hasn't published shows `--` and never a sample number.
- **No error rate during a call.** There is no answer key during a call, so the end screen shows how often each
  reader alone **disagreed** with the final word, labelled that way.
- **The status line comes from what is actually arriving.** It checks how recently each part published, and whether
  the bus logger (`buslog.record`) is writing.

| On screen | From |
| --- | --- |
| lips to voice | first `dec.lock` word time to `timing.voice` `sonic_first_audio` (before the phone network) |
| fixed by Claude, asked to repeat | `claude.repair` edits, `dec.repeat` |
| Judge bubble | `partner.text` (partial while speaking, then final) |
| Austin's words and their colours | `dec.hyp` `locked` / `unlocked` / `source`, plus the struck word from `claude.repair` |
| Candidate bars and the lock threshold | `dec.hyp` `cands`, `threshold` (optional, below) |
| Camera | `cam.preview` (optional, below); without it, the face mesh from `cam.face` |
| Emotion chip, "Sounds warm" | `cam.tick` `emotion` (the expression model) |
| Pads | `emg.raw`: level (RMS against a running peak), contact (signal not flat, not on a rail) |
| Latency | `cam.tick` `timing_ms.total`, `timing.dec` (`lock`, `window`), `timing.claude`, `timing.voice` |

## Optional fields the screen uses when they arrive

These are in docs/TEAM_TASKS_AND_INTERFACES.md; the screen works without them and shows more with them.

- `cam.preview` (camera reader): `t`, `w`, `h`, `jpeg`, about 10 fps, 640 px wide. Mirrors the webcam, which only
  the lip reader can open, with `pts`: that frame's own face points (x, y, z), so the pads are placed exactly;
  without them the dashboard takes the `cam.face` closest in time (within 60 ms).
- `dec.hyp` `cands` [{`w`, `cam`, `emg`, `claude`}] (top 3 for the newest word) and `threshold` (decoder).
- `dec.lock` words `cam_top`, `emg_top`: each reader's own pick for that word, for the end screen's disagreement
  bars (decoder).
- `timing.dec` `{stage: 'lock' | 'window', ms}` (decoder).

## Test

```bash
venv-call/bin/python -m pytest tests/dashboard -q -c tests/dashboard/pytest.ini
venv-call/bin/python -m dashboard.demo_feed --fixtures /tmp/fx    # one snapshot per state, no bus
```

`demo_feed.simulate()` feeds the scripted call into `CallState` on a simulated clock, so the tests check every
state in the design without sockets or timing. `?offline` opens the page without the websocket;
`window.reclaimRender(snapshot)` draws a given snapshot, which is how the screenshots are made.
