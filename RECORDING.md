# Recording

How we collect training data for quick phrases.
Ported from the old `silent-voice-rig` repo and updated for 4 channels on Austin's laptop.

## What a session is

One wearing of fresh pads, during which Austin silently mouths the phrases while Thai runs the recorder.
A pause with the pads still on is the same session, and a new session means fresh pads.
We do 3 sessions: a pilot Friday at 11:45pm, then 11:00am and 7:15pm Saturday, all at the venue.
The pilot only goes full if the recorder saves clean clips in the first 10 min, and otherwise it stops at 5 clips per phrase and sessions 2 and 3 record the rest.
Each session is about 15 min of setup and 15 to 20 min of mouthing.

Across the 3 sessions we need 50 clips of each of the [10 phrases](PHRASES.md) plus 50 rest clips, so each session records about 185 clips.

## Why it's set up this way

- **3 sessions:** pads land in slightly different spots each time, so varied sessions teach the model to cope with that shift.
- **Shuffled order:** the model can't learn drift over a session, like pads drying out, instead of the phrase.
- **Rest clips:** the classifier learns to output nothing when nobody is mouthing.
- **No sound and no breath, not a whisper:** a laryngectomy patient can't push air through the mouth.
- **No video:** quick phrases uses no camera, so only the EMG channels are saved.

## Before

- [ ] The [10 phrases](PHRASES.md) are locked
- [ ] `rig/uno` is on the Uno, and the recorder runs and saves a test clip
- [ ] 12 pads (4 boards × 3), alcohol prep pads, and a skin marker, plus 12 spares
- [ ] Austin's laptop is on battery with the charger unplugged, unless a USB isolator sits before the Uno, and phones are away from the cables
- [ ] A chair and a quiet corner, plus water for Austin
- [ ] The training script is ready to load a new session

## Steps

1. Wipe the pad spots with alcohol and let them dry.
2. Put the pads on the [marked spots](HARDWARE.md#pad-sites), with each board's REF on bare bone behind the ear.
3. Run the [signal check](HARDWARE.md#signal-check).
   Every channel must say OK, so fix any other verdict now, not after recording.
4. Record.
   The recorder shows one phrase at a time in shuffled order, with rest mixed in.
   For each clip, Austin stays still for 1 s, mouths the phrase once, then stays still.
5. If a clip goes wrong, like a cough, a laugh, or the wrong phrase, press redo instead of keeping it.
6. Check that `labels.csv` matches the files, and start training on the new session.

```
recordings/
  labels.csv              file, phrase, session, speaker
  session_01/
    meta.json             date, speaker, sample rate, channel to site
    0001.npy              raw readings, shape (samples, 4), unfiltered
  session_02/
```

The phrase recorder isn't built yet.
It should read through `rig/reader.py` and save raw clips, and training runs `rig/filters.py` over each clip in order.

## Success

- Every channel said OK in the signal check.
- About 185 clips saved, with at least 15 of each phrase and rest.
- No channel sits flat or pinned at 0 or 1023 for a whole clip.
- Rest clips look quiet next to phrase clips.
- The session loads and trains without errors.
