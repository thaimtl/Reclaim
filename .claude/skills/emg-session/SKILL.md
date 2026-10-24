---
name: emg-session
description: Record and check EMG signals from the Reclaim pad rig. Use when someone wants to test channels, check pad contact or hum, record training data, or asks whether the signal looks good.
---

# EMG session

The rig is 4 BioAmp boards on A0 to A3 of an Uno running `rig/uno`, read by `rig/reader.py` and filtered by `rig/filters.py`.
Read HARDWARE.md for pad sites and wiring.

## Before recording

1. Set up once: `python3 -m venv rig/.venv && rig/.venv/bin/pip install -r rig/requirements.txt`.
   On Windows the interpreter is `rig\.venv\Scripts\python`.
2. Confirm the laptop is on battery and nothing from wall power touches it, unless a USB isolator sits before the Uno.
3. Find the Uno: `rig/.venv/bin/python -m serial.tools.list_ports`.
   No port means the USB is unplugged or the cable only carries power.
4. If the port is busy, the Arduino IDE's Serial Monitor or Plotter holds it, even with the window closed.
   Ask the user to quit the IDE fully, don't kill its process.
5. If the reader says "no frames from the Uno", the Uno runs another sketch, so flash `rig/uno`:
   `arduino-cli compile --upload -p <port> --fqbn arduino:avr:uno rig/uno`.
   On macOS, the IDE bundles it at `/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli`.

## Recording

`rig/session.py` waits for Enter, so Claude can't drive it.
Ask the user to run `rig/.venv/bin/python rig/session.py <name>.npz` in their own terminal, then say "done".
It speaks each step, shows live filtered bars, and saves raw data plus a step label per sample.
Tell them the pace before starting: one syllable or phrase per second, with a short pause between.
To test other movements, edit `STEPS` in `rig/session.py`, and keep `rest` steps between tasks.

## Reading it

Run `rig/.venv/bin/python rig/check.py <name>.npz`, and add `--windows` to see every 0.25 s.

| Column | Meaning | Good |
|---|---|---|
| clipped | Share of samples at 0 or 1023, which the filter can't recover | 0% |
| hum | Raw spread at rest, almost all 60 Hz mains hum | Under 20, the filter removes the rest |
| rest | Filtered level at rest | About 1 |
| task columns | 90th percentile burst level divided by rest | 2x to 6x |

Then check `--windows`: each syllable should be its own burst, about one a second, with quiet gaps.
The thresholds come from Austin's sessions on October 23, 2026, so treat them as a guide.

| Verdict | Likely cause | Fix |
|---|---|---|
| MAXES OUT | Loose pad or REF, often hair under the REF | Re-press snaps, fresh pads, REF on bare bone |
| NO RESPONSE with low hum | IN+ and IN− bridged by gel, or cable not seated | Separate the pads, push the cable into the JST |
| Pure hum and no bursts | REF floating, or a charger plugged in | Press REF down, unplug the charger |
| WEAK | Light contact | Press pads and REF down, wipe skin with alcohol |

The Uno's reset button restarts the sketch and never fixes contact.
To tell a bad board from bad pads, swap that channel's electrode cable onto a board that passes, and see whether the problem moves.

## Training data

Keep recordings raw.
Run `EmgFilter` over each raw recording in order, the same way it runs live, and never use `filtfilt` or other offline filters.
