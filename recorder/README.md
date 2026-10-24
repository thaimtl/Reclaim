# Recorder

`recorder/record.py` records the three sessions in [RECORDING.md](../RECORDING.md): the webcam and the 4 EMG channels together, one clip per sentence, with the sentences from [docs/recording/](../docs/recording/). It runs on Austin's laptop, on battery.

## Set up once

From the repo root, in a terminal. The recorder shares the rig's virtual environment.

macOS or Linux:

```bash
python3 -m venv rig/.venv
rig/.venv/bin/pip install -r recorder/requirements.txt
rig/.venv/bin/python recorder/test_record.py        # prints "ok"; needs no hardware
```

Windows, in PowerShell:

```powershell
py -m venv rig\.venv
rig\.venv\Scripts\pip install -r recorder\requirements.txt
rig\.venv\Scripts\python recorder\test_record.py
```

Flash the Uno with `rig/uno/uno.ino`. In Arduino IDE: open the file, pick the board (Arduino Uno) and its port, press Upload. Then **quit Arduino IDE completely**, since it holds the serial port even with its windows closed.

## Each session

1. Unplug the laptop's charger. The pads connect to the laptop through the Uno, so it runs on battery whenever they're on skin; the recorder refuses to start a clip while the charger is in.
2. Put the pads on, then run the signal check until every channel says OK ([HARDWARE.md](../HARDWARE.md#signal-check)).
3. Start the recorder:

    ```bash
    rig/.venv/bin/python recorder/record.py --session 1
    ```

    On Windows: `rig\.venv\Scripts\python recorder\record.py --session 1`. Use `--session 2` and `--session 3` for the others.

    It finds the Uno by itself and uses camera 0. If that's the wrong camera, run `--list-cameras` and pass `--camera 1`. If the Uno isn't found, pass `--port COM3` (Windows) or `--port /dev/cu.usbmodem...` (macOS).
4. Click the window so it gets the keys, then for each sentence:

    | Key | Does |
    | --- | --- |
    | SPACE | Start the clip, then SPACE again to stop it |
    | R | Redo the last clip: deletes it and asks for it again |
    | S | Skip this prompt for now (it comes back next run) |
    | H | Show or hide the key help |
    | Q or ESC | Quit. Everything saved so far stays; run the same command to resume |

    Press SPACE, stay still for half a second, mouth the sentence once (mouth open, no sound) at the pace shown, stay still half a second, press SPACE. Rest clips show a cue instead of a sentence (swallow, nod, smile...): do it during the clip.

The window shows the mirrored camera with an oval where the mouth should sit, the 4 EMG levels (red means the channel hits 0 or 1023: press the pads or REF down), the camera's frame rate, dropped EMG frames, progress, and warnings: charger plugged in, EMG silent, camera stopped. After each clip it shows the clip's own warnings, such as short EMG or a flat channel; redo the clip if one appears.

Order: about 260 clips per training session in a fixed shuffled order, with sentences at natural pace, sentences at a faster pace (marked FAST PACE), and 10 rest clips mixed in. Session 3 is the test set, at demo pace.

## What it saves

Under `$RECLAIM_WORKDIR/recordings/session_0N/`, or `recordings/` in the repo if `RECLAIM_WORKDIR` isn't set (git ignores it). The layout is the one `svr/recordings.py` loads, so fine-tuning reads it directly.

| File | Contents |
| --- | --- |
| `<id>.mp4` | Camera video, unmirrored, at the camera's resolution (default 1280×720) |
| `<id>_video_ts.npy` | Laptop-clock time (`time.monotonic()`) of every video frame |
| `<id>_emg.npy` | Raw Uno readings, shape (samples, 4), uint16, unfiltered; columns A0 to A3 = sites 3, 2, 6, 1 |
| `<id>_emg_ts.npy` | Laptop-clock time of every EMG sample |
| `manifest.jsonl` | One line per clip: id, set (C training, E test, B rest), prompt, pace, file names, clip start and end, sample and frame counts, warnings |
| `meta.json` | The session: speaker, sample rate, pins and sites, camera settings, recorder version |

The EMG is saved raw on purpose: training runs `rig/filters.py` over each clip in order, the same filter as the live system.

## How camera and EMG stay in sync

Both use the laptop's `time.monotonic()`. Video frames are stamped right after the camera hands them over. EMG samples come in USB chunks, so each chunk's arrival is paired with the running sample count, and a straight line through the earliest arrivals (the ones with the least USB delay) gives every sample's time. That handles the Uno's clock running a bit fast or slow. The test checks it stays within 3 ms with 1 to 15 ms of USB jitter and a 0.5% clock error. Dropped serial frames (from the firmware's counter) are filled with the last reading and counted, so later samples keep their right time.

## Trying it without hardware

```bash
rig/.venv/bin/python recorder/record.py --session 1 --camera fake --fake-emg --allow-charger --root /tmp/rec
```

`--camera` also takes a video file. `--allow-charger` only works with `--fake-emg`.

## Notes for changing it (people and Claude)

- **Prompts:** sentences come from `docs/recording/session{1,2}.txt` and `session3_test.txt` (`id<TAB>sentence`), made by `docs/recording/make_prompts.py`. Sentences numbered 201 and up in sessions 1 and 2 are FAST PACE (`FAST_FROM`); the rest cues are `REST_CUES`. The shuffle is seeded by the session number, so a resumed session keeps its order.
- **The file format is a contract** with `svr/recordings.py` (`load_manifest`, `load_emg`). Change both together, and tell the other sessions on the team board.
- **EMG input** goes through `rig/reader.py` (`frames(port, stats)`), Thai's reader for the `rig/uno` firmware. Channel count, pins and sites live in `rig/` and in `CHANNELS` and `SITES` here.
- **Threads:** the camera grabs and stamps frames; a writer thread encodes MP4 so a slow disk never stalls capture (a full queue drops and counts a frame); the EMG thread reads the Uno. The window and keys run in the main thread (OpenCV).
- **Tests:** `recorder/test_record.py` runs everything with a fake camera and fake EMG. Run it after any change.
