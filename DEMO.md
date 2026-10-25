# Running the demo

One launcher starts everything in order on Austin's laptop and checks each part is working before it starts the next: [`demo/launch.py`](demo/launch.py).

## Demo day

1. Laptop **on battery** (unplug the charger before the pads go on: there is no USB isolator).
2. Pads on, then the signal check (`.claude/skills/signal-check`).
3. Double-click **`Launch Silent Voice.bat`** in the repo folder.
   - It asks whether the signal check is done; press Enter.
   - It runs the pre-flight checks, then starts each part. Every line says `[ ok ]` or `[FAIL]` with the fix.
   - When the decoder starts, it shows a banner: **Austin mouths a few normal sentences for 60 s** (EMG calibration).
4. When it shows **SILENT VOICE IS READY**, the dashboard is open in the browser and the judge can call the number on screen.
5. At the end, press **Ctrl-C** in the launcher window. Everything stops in order, and the logs are kept in the folder it printed.

Each start points the Twilio number at the tunnel of that run, so there is nothing to set by hand, and the number stays the same.

## If something is red

| What it says | What to do |
| --- | --- |
| `keys in call/keys.env: missing ...` | Get `call/keys.env` from Frank (never commit it). |
| `venv-... in ...` | The environment is missing, or `RECLAIM_WORKDIR` points at the wrong folder (one-time setup below). |
| `ports free: [...] in use` | Another launcher, or a part from an earlier run, is still open: close it, or restart the laptop. |
| `webcam 0` | Plug in the C920, close apps that use the camera (Zoom, Camera, Teams), or start with `--cam 1`. |
| `Uno on ?` | Plug in the Uno, or pass `--port COM5` (Device Manager, Ports). No pads today: `--no-emg`. |
| `the laptop is charging` | Unplug the charger. |
| `cloudflared` | Install it (one-time setup). |
| `... did not come up` | The lines under it are that part's own output; its full log is in the run folder. |
| `does not reach the call server` | Internet: check the Wi-Fi, then start again. |
| `Twilio webhook not updated` | Internet, or the Twilio keys in `call/keys.env`. |
| A part stops while running | It is restarted by itself (up to 3 times). If it was the call server, the judge calls again. |

Fallbacks, from a terminal in the repo folder (`"Launch Silent Voice.bat" --no-emg`):
- `--no-emg`: pads off, camera + Claude.
- `--no-claude`: without Claude.
- `--cam 1`: another webcam. `--camera-settings`: lock exposure and focus in a dialog first.
- `--check`: only the pre-flight checks.
- `--rehearse`: no camera, no pads: call the number and session 3 plays into the call, a dry run of everything after the camera.

## One-time setup

Done once per laptop.

1. `RECLAIM_WORKDIR`: the folder outside the repo that holds the data and environments (README, "Code and data"). The `.bat` also finds `venv-call` next to the repo by itself.
2. The three environments in it, with [uv](https://docs.astral.sh/uv/):
   - `venv-call` (call, dashboard, Claude, logger, launcher): `uv venv --python 3.11 venv-call`, then `uv pip install --python venv-call\Scripts\python.exe -r requirements-call.txt`
   - `venv-live` (camera reader): see the top of [`requirements-camera.txt`](requirements-camera.txt).
   - `venv-svr` (decoder, EMG publisher): see the top of [`requirements-fusion.txt`](requirements-fusion.txt).
3. `call/keys.env` from Frank: `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_NUMBER`. It is git-ignored: never commit it.
4. `cloudflared`: `winget install --id Cloudflare.cloudflared`, or put `cloudflared.exe` in `RECLAIM_WORKDIR`.
5. Data under `RECLAIM_WORKDIR`: the SVR-1 checkpoint, the language model, the lexicon and the fitted decoder (listed at the top of [`fusion/run.py`](fusion/run.py)), and for `--rehearse` the camera log `logs/cam_s3_test`.
6. Test it: `"Launch Silent Voice.bat" --check`. Every line should be `[ ok ]`.

## What it starts

In this order, each one only once the one before is ready:

| Part | Command | Ready when |
| --- | --- | --- |
| Logger | `buslog.record` into the run folder | it logs |
| EMG | `rig.publish COMx` | `emg.raw` arrives on the bus |
| Camera | `lipread.run --cam 0` | the first `cam.preview` frame arrives |
| Claude | `claude_service.service` | it has its key |
| Decoder | `fusion.run` (`--calibrate 60` with pads) | it runs (after the calibration) |
| Tunnel | `cloudflared` to the call server's port | it has a public address |
| Call | `call.server`, told the tunnel's address | Sonic is connected; then the address is checked from outside and the Twilio number pointed at it |
| Dashboard | `dashboard.server` | its page loads |

Call-audio recording is always off. No key is ever printed or logged by the launcher.
