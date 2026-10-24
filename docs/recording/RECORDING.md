# Recording sessions

Three sessions. Sessions 1 and 2 are training data; session 3 is the test set and is never trained on. Take the pads off and put them back between sessions (same skin marks), so the model learns to handle small shifts in placement.

| File | Session | Sentences | Use |
| --- | --- | --- | --- |
| `session1.txt` | 1 (now) | 250 | Training |
| `session2.txt` | 2 | 250 | Training |
| `session3_test.txt` | 3, last, in demo conditions | 100 | Test only, never trained on |

Each line is `id <tab> sentence`. Use the id as the utterance id in the recording manifest (`svr/recordings.py` format), with set `C` for sessions 1 and 2 and set `E` for session 3.

## Before each session

1. Wipe the skin with alcohol and put the pads on the marked spots: sites 1, 2, 3, 5, 6, 8, one REF per board behind the ears.
2. Check that every channel bursts when you mouth "ta ta ta, ka ka ka, pa pa pa".
3. Camera and EMG record on the same laptop with the same clock (`time.monotonic()`), face lit from the front, the camera at the height and distance you'll use in the demo.
4. Electrical safety: a USB isolator between the Uno and the laptop, or the laptop on battery.

## For every sentence

1. Read the sentence on screen.
2. Start recording, wait half a second, and mouth the sentence: mouth open, no sound, lips, tongue and jaw moving as if speaking.
3. Wait half a second, then stop.
4. If you stumble, redo it under the same id.

Pace: natural conversational speed for sentences 1 to 200 of sessions 1 and 2, then faster than natural for sentences 201 to 250. Session 3 is all at the pace you'll use in the demo call.

## Rest clips, about 2 minutes per session

Ten 3-second clips labelled as set `B`, no prompt: still face, swallow, nod, look left, look right, smile, raise eyebrows, take a breath, lick your lips, small cough. These teach the model what is not speech.

## Time

About 6 to 7 seconds per sentence including reading and starting, so sessions 1 and 2 take about 30 minutes each, and session 3 about 12 minutes. Do session 3 last, as close to the demo as you can, in the demo room.
