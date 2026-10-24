# Recording

How we collect the data SVR-1, the EMG model, is fine-tuned on: Austin silently mouthing varied sentences, with the camera and the 4 EMG channels recording together on one clock. The recorder steps through the sentences, records each one, and saves everything in the layout below. Nothing here is the pitch: the judge can ask anything, so we record varied conversational sentences.

## What a session is

One wearing of fresh pads, during which Austin mouths the sentences while someone runs the recorder.
A pause with the pads still on is the same session, and a new session means fresh pads.

| Session | Sentences | Use |
| --- | --- | --- |
| 1 | 250 ([docs/recording/session1.txt](docs/recording/session1.txt)) | Training |
| 2 | 250 ([docs/recording/session2.txt](docs/recording/session2.txt)) | Training |
| 3, last, in demo conditions | 100 ([docs/recording/session3_test.txt](docs/recording/session3_test.txt)) | Test only, never trained on |

Each session also records 10 rest clips. Sessions 1 and 2 take about 15 minutes of setup and 30 minutes of mouthing; session 3 about 12 minutes of mouthing.

Some sentences come with a delivery to hold, shown in amber above the sentence. The rest are mouthed plainly, with a relaxed face.

| Delivery | Sessions 1 and 2, each | Session 3 | What Austin does |
| --- | --- | --- | --- |
| Slow | 20 | | Mouths it at about half his natural speed |
| Smile | 10 | 10 | Holds a real smile for the whole sentence |
| Frown | 10 | 10 | Looks upset or annoyed for the whole sentence |
| Surprised | 10 | | Brows up, eyes wide, for the whole sentence |
| Soft | 10 | | Small, relaxed movements, as if whispering |
| Emphatic | 10 | | Big, firm movements, as if speaking up |

Only natural-pace sentences get a delivery (demo-pace in session 3), at most one each; the fast ones stay plain. That leaves 130 plain natural-pace and 50 fast sentences per training session.

## Why it's set up this way

- **3 sessions:** pads land in slightly different spots each time, so varied sessions teach the model to cope with that shift, and session 3 checks it on a placement the model has never seen.
- **Varied sentences, not a script:** the demo is a live call where the judge can ask anything.
- **Shuffled order:** the recorder mixes sentences, fast-pace sentences and rest clips, so the model can't learn drift over a session, like pads drying out, instead of the words.
- **Rest clips:** the model learns what is not speech (still face, swallowing, nodding).
- **Deliveries:** the voice follows Austin's expression, pace and effort ([README](README.md#how-it-works)), so the demo has him smile or frown while he mouths and speak slowly, softly or emphatically. A smile changes the EMG too (site 1 sits near the smile muscles), so SVR-1 has to see it in training; the slow, soft and emphatic sentences cover the range its pace and effort cover live; and session 3's smiling and frowning sentences test it, and give the emotion tagger Austin's real expressions.
- **Mouth open, no sound and no breath:** the way the demo is done, and what a laryngectomy patient can do.
- **Camera and EMG together:** the fusion decoder and the locking model are tuned on recordings where the camera's reading, the EMG reading and the true sentence are all known.

## Before

- [ ] `rig/uno` is on the Uno, Arduino IDE is fully quit, and `recorder/test_record.py` prints ok
- [ ] 12 pads (4 boards × 3), alcohol prep pads, and a skin marker, plus 12 spares
- [ ] Austin's laptop on battery with the charger unplugged, and no other cable from wall power on the laptop
- [ ] The webcam at the height and distance used in the demo, the face lit from the front, nothing bright behind Austin
- [ ] A chair and a quiet corner, plus water for Austin

## Steps

1. Wipe the pad spots with alcohol and let them dry.
2. Put the pads on the [marked spots](HARDWARE.md#pad-sites), with each board's REF on bare bone behind the ear.
3. Run the [signal check](HARDWARE.md#signal-check). Every channel must say OK, so fix any other verdict now, not after recording.
4. Start the recorder: `rig/.venv/bin/python recorder/record.py --session 1` (Windows: `rig\.venv\Scripts\python recorder\record.py --session 1`). Setup and keys: [recorder/README.md](recorder/README.md). It shows one sentence at a time and refuses to record while the charger is plugged in.
5. For each sentence: read it and the amber delivery line if there is one, press SPACE, stay still for half a second with a relaxed face, mouth the sentence once at the pace shown (natural, fast when marked FAST PACE, slow when marked SLOW PACE) and in the delivery shown (hold the smile, frown or surprise for the whole sentence), stay still for half a second, press SPACE.
6. If a clip goes wrong, like a stumble, a cough, a laugh or a missed word, or the recorder shows a warning for it, press R to redo it.
7. Quit with Q at any time: everything saved stays, and the same command resumes where you left off.

## What gets saved

The recorder writes the layout `svr/recordings.py` loads, so fine-tuning reads it directly. Details: [recorder/README.md](recorder/README.md).

```
recordings/                  ($RECLAIM_WORKDIR/recordings; not committed to git)
  session_01/
    manifest.jsonl           one line per clip: id, set (C training, E test, B rest), prompt, pace, expression, effort, file names,
                             clip start and end on the laptop clock, sample and frame counts, warnings
    meta.json                speaker, sample rate, pins and sites (A0 to A3 = sites 3, 2, 6, 1), camera settings
    s1_001.mp4               camera video, unmirrored
    s1_001_video_ts.npy      laptop clock (time.monotonic) of each video frame
    s1_001_emg.npy           raw readings, shape (samples, 4), uint16, unfiltered
    s1_001_emg_ts.npy        laptop clock of each EMG sample
  session_02/
```

The EMG stays raw: training runs `rig/filters.py` over each clip in order, the same filter the live system uses. The video and the EMG share the laptop's clock, so each frame can be matched to its samples to about a millisecond.

## Success

- Every channel said OK in the signal check.
- Every sentence of the session saved, plus the rest clips, each with video and EMG.
- No channel sits flat or pinned at 0 or 1023 for a whole clip.
- No dropped EMG frames, and the video runs at a steady frame rate.
- Rest clips look quiet next to sentence clips.
- Every delivery in the table recorded as shown: if Austin forgot to hold the smile or went at the wrong speed, press R.
- The session loads in the training script without errors.
