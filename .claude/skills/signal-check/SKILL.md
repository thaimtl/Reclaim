---
name: signal-check
description: Run and read the pre-recording signal check on the Reclaim pad rig, on macOS or Windows. Use when someone wants to test channels, check pad contact or hum, or asks whether the signal looks good. It is QA before data collection, not data collection.
---

# Signal check

The rig is 4 BioAmp boards on A0 to A3, which are sites 3, 2, 6, and 1 in Gaddy and Klein 2020, on an Uno running `rig/uno`, read by `rig/reader.py` and filtered by `rig/filters.py`.
Read HARDWARE.md for pad sites and wiring, and RECORDING.md for real data collection.
This check runs after wiring and every time pads go on, and it never produces training data.

## Paths per OS

| | macOS | Windows |
|---|---|---|
| Python in the venv | `rig/.venv/bin/python` | `rig\.venv\Scripts\python` |
| Create the venv | `python3 -m venv rig/.venv` | `py -m venv rig\.venv` |
| Uno port | `/dev/cu.usbmodem…` | `COM3` or similar |
| Bundled arduino-cli | `/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli` | Usually `C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe` |

Below, `PY` means the venv Python for the current OS.
The Windows arduino-cli path has not been checked on a real machine.

## Before the check

1. Set up once: create the venv, then `PY -m pip install -r rig/requirements.txt`.
2. Confirm the laptop is on battery and nothing from wall power touches it. We have no USB isolator, so there is no exception.
3. Find the Uno with `PY -m serial.tools.list_ports`.
   No port means the USB is unplugged or the cable only carries power.
   On Windows, a clone Uno with a CH340 chip also needs the CH340 driver.
4. If the port is busy, the Arduino IDE's Serial Monitor or Plotter holds it, even with the window closed.
   Ask the user to quit the IDE fully, don't kill its process.
5. If the reader says "no frames from the Uno", the Uno runs another sketch, so flash `rig/uno`:
   `arduino-cli compile --upload -p <port> --fqbn arduino:avr:uno rig/uno`.

## Running it

Claude runs it, and the wearer follows the spoken cues.
1. Tell the user the steps and the pace: one syllable or phrase per second, with a short pause between.
2. Ask them to say "ready" once pads are on and the wearer is relaxed.
3. Run `echo | PY rig/signal_check.py check.npz` with a 120 s timeout, where `echo |` presses Enter for the script.
It speaks each step, with `say` on macOS and the built-in speech engine on Windows, and saves raw data plus a step label per sample.
The live bars only show when a person runs it in their own terminal, which also works if the user prefers.
To test other movements, edit `STEPS` in `rig/signal_check.py`, and keep `rest` steps between tasks.

## Reading it

Run `PY rig/signal_report.py check.npz`, and add `--detail` to see every 0.25 s.

| Column | Meaning | Good |
|---|---|---|
| clipped | Share of samples at 0 or 1023, which the filter can't recover | 0% |
| hum | Raw spread at rest, almost all 60 Hz mains hum | Under 20, the filter removes the rest |
| rest | Filtered level at rest | About 1 |
| task columns | 90th percentile burst level divided by rest | 2x to 6x |

Then check `--detail`: each syllable should be its own burst, about one a second, with quiet gaps.
The thresholds come from Austin's sessions on October 23, 2026, so treat them as a guide.

| Verdict | Likely cause | Fix |
|---|---|---|
| MAXES OUT | Loose pad or REF, often hair under the REF | Re-press snaps, fresh pads, REF on bare bone |
| NO RESPONSE with low hum | IN+ and IN− bridged by gel, or cable not seated | Separate the pads, push the cable into the JST |
| Pure hum and no bursts | REF floating, or a charger plugged in | Press REF down, unplug the charger |
| WEAK | Light contact | Press pads and REF down, wipe skin with alcohol |

The Uno's reset button restarts the sketch and never fixes contact.
To tell a bad board from bad pads, swap that channel's electrode cable onto a board that passes, and see whether the problem moves.
