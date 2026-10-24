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

## Why it's set up this way

- **3 sessions:** pads land in slightly different spots each time, so varied sessions teach the model to cope with that shift, and session 3 checks it on a placement the model has never seen.
- **Varied sentences, not a script:** the demo is a live call where the judge can ask anything.
- **Shuffled order:** the recorder mixes sentences, fast-pace sentences and rest clips, so the model can't learn drift over a session, like pads drying out, instead of the words.
- **Rest clips:** the model learns what is not speech (still face, swallowing, nodding).
- **Mouth open, no sound and no breath:** the way the demo is done, and what a laryngectomy patient can do.
- **Camera and EMG together:** the fusion decoder and the locking model are tuned on recordings where the camera's reading, the EMG reading and the true sentence are all known.

## Before

- [ ] `rig/uno` is on the Uno, and the recorder runs and saves a test clip
- [ ] 12 pads (4 boards × 3), alcohol prep pads, and a skin marker, plus 12 spares
- [ ] Austin's laptop on battery with the charger unplugged, and no other cable from wall power on the laptop
- [ ] The webcam at the height and distance used in the demo, the face lit from the front, nothing bright behind Austin
- [ ] A chair and a quiet corner, plus water for Austin

## Steps

1. Wipe the pad spots with alcohol and let them dry.
2. Put the pads on the [marked spots](HARDWARE.md#pad-sites), with each board's REF on bare bone behind the ear.
3. Run the [signal check](HARDWARE.md#signal-check). Every channel must say OK, so fix any other verdict now, not after recording.
4. Start the recorder for the session (instructions in `recorder/README.md`). It shows one sentence at a time.
5. For each sentence: read it, start the clip, stay still for half a second, mouth the sentence once at the pace shown (natural, or fast for the sentences marked fast), stay still for half a second, stop the clip.
6. If a clip goes wrong, like a stumble, a cough, a laugh or a missed word, redo it instead of keeping it.
7. At the end, the recorder checks that every clip has both video and EMG and that `labels.csv` matches the files.

## What gets saved

```
recordings/
  labels.csv                 id, session, set, pace, sentence, emg, video, speaker
  session_01/
    meta.json                date, speaker, sample rate, channel to site (A0 to A3 = sites 3, 2, 6, 1),
                             camera resolution and frame rate, dropped frames
    s1_001.emg.npy           raw readings, shape (samples, 4), unfiltered
    s1_001.emg_t.npy         laptop clock (time.monotonic) of each EMG sample
    s1_001.mp4               camera video
    s1_001.video_t.npy       laptop clock of each video frame
  session_02/
```

The EMG stays raw: training runs `rig/filters.py` over each clip in order, the same filter the live system uses. The video and the EMG share the laptop's clock, so each frame can be matched to its samples exactly. `recordings/` is not committed to git.

## Success

- Every channel said OK in the signal check.
- Every sentence of the session saved, plus the rest clips, each with video and EMG.
- No channel sits flat or pinned at 0 or 1023 for a whole clip.
- No dropped EMG frames, and the video runs at a steady frame rate.
- Rest clips look quiet next to sentence clips.
- The session loads in the training script without errors.
