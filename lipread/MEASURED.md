# Camera reader: measured numbers

Every number measured for the live camera reader, kept from the planning log (the old `plans/live-avsr/CHANGELOG.md`).
Unless a line says otherwise it was measured on the dev box (RTX 5090, WSL2, shared GPU), not on the demo laptop
(RTX 5060); laptop numbers come from `bash lipread/bench/laptop_check.sh` and `python -m lipread.bench.tick_bench`. Plan:
`docs/EVAN_REALTIME_LIP_READING.md`. Superseded design notes (WebSocket events, 200 ms ticks, the old `plans/`) are
history; the code and `bus_schema.md` are current.


Since this log was written: `live/` became `lipread/` plus the team bus `bus/`; `laptop_check.sh` became
`lipread/bench/laptop_check.py` (Windows and Linux); `record.py` and `commit.py` were removed (the team's `recorder/`
records, and fusion locks words); the opt-in EmotiEffLib classifier (`emotion_net.py`), `smooth_check.py` and
`prompts.txt` were removed (the blendshape tagger is the plan's emotion, and k0 smoothing was chosen); the bus moved from `ipc://` to `tcp://127.0.0.1:5556` with `bus.now()`
(`time.perf_counter()`) as the clock. Paths below are the new ones; the numbers are unchanged.

All numbers measured on the dev box (RTX 5090, Ryzen 7 9850X3D, WSL2), NOT the demo laptop (RTX 5060 8 GB).
Laptop numbers come from `lipread/bench/laptop_check.sh` (T1).

### T0, 2026-10-24 ~03:40 UTC (estimate was ~1 h; took ~10 min of wall time)

- `venv-live`: Python 3.11.17, torch 2.11.0+cu130, mediapipe 0.10.21, numpy 1.26.4, websockets 17.3, av, jiwer, pytest.
  Install 4 min 10 s. `import mediapipe, torch, cv2` OK; CUDA 4096^2 matmul OK. Note: mediapipe pulls
  `opencv-contrib-python` 4.11.0.86, which shadows the pinned `opencv-python` 4.10 (`cv2.__version__` = 4.11.0).
- Weights: HF part of `camera/download_weights.sh` (vsr 1.0 GB, LM 215 MB, spm, face landmarker), 68 s. Sample:
  `third_party/lipflow/samples/2016-03-12.mov` (Wikimedia, as Lipflow's `--samples`), segment 20.4-28.1 s.
- Sample: 231 source frames -> 192 model frames (25 fps), face found in 231/231.

| Stage (192 frames = 7.7 s unless noted) | Measured (5090) | Plan estimate |
| --- | --- | --- |
| Landmarks (MediaPipe, full 640x360 frame, incl. video decode) | 6.4 ms/frame (CPU) | - |
| Align + crop (`mouth_rois`) | 0.21 ms/frame | - |
| Encoder fp32 | 13.0 ms median (max 25.6); 250 frames 13.7 ms | "tens of ms on the 5060" |
| Encoder fp16 (autocast) | 15.7 ms; 250 frames 17.3 ms (slower than fp32 here) | fp16 = speed-up (D15) |
| CTC greedy | 0.18 ms | ~1 ms (Lipflow comment) |
| Beam 10 + LM | 1459 ms (96 frames: 697 ms) | "beam 5 if > 1 s on the laptop" (D11) |
| Beam 5 + LM | 1141 ms (96 frames: 549 ms) | - |
| Beam 3 + LM | 1105 ms (96 frames: 498 ms) | live-decoder candidate (D8) |
| Beam 3, no LM | 904 ms (96 frames: 305 ms) | - |
| Peak VRAM (model + LM + beam 10) | 1.58 GiB | < 8 GB |
| Load LipReader | 6.5 s | - |

Transcripts (Lipflow truth: BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE;
the 20.4-28.1 s segment also contains "in 1943", which the truth omits):
- greedy (fp32 and fp16 identical): BORN IN NEW YORK CITY IN RACE MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH
  COLLEGE IN ONE NINE HUNDRED FORTY THREE (2 errors in the 16 truth words)
- beam 10 + LM: ... AND RAISED MOSTLY ... IN NINETEEN FORTY THREE (16/16 truth words)

Findings:
1. Beam time barely depends on beam size (10: 1.46 s, 3: 1.11 s): per-step overhead dominates, ~linear in clip
   length. Beam 3 as a live decoder (D8) cannot fit an 80-160 ms update even on the 5090: greedy is the only live
   decoder (cut list item 3). D11's "beam 5 if > 1 s" saves only ~0.3 s.
2. Encoder is ~13 ms whatever the length (launch-bound on the 5090); fp16 autocast is slower here. Laptop decides.
3. fp16 depends on how: whole encoder `.half()` = 9.2 ms at 250 frames vs fp32 12.9 ms (lane C, `laptop_check.sh`
   section 10); autocast = 17.3 ms (slower). `lipread/model.py` uses `.half()`, behind the fp16-vs-fp32 test.

### Lane B (logic + events), accepted 2026-10-24 ~04:00 UTC

- `tests/lipread/test_commit.py`, `test_endpoint.py`, `test_events.py`: 37 passed in 0.58 s (lead re-run).
- `lipread.stub`: 76/76 events schema-valid over ~9 s to a real websockets client (lane B run).
- `lipread/bench/prompts.txt`: 30 tune + 20 heldout, unique, 4-9 words.

### Lane C (camera + laptop), accepted 2026-10-24 ~04:10 UTC

- `tests/lipread/test_crop.py`: 12 passed (whole `tests/lipread`: 49 passed in 20.2 s, lead re-run).
- LiveCropper k = 6 symmetric vs Lipflow `mouth_rois`, 192 frames: identical (max diff 0). Default window (past 6 +
  6 ahead): mean 0.099, 97.9% of frames <= 1.0 (differences only in the first/last 6 frames). k = 2: mean 2.0, max 6.8.
- Replay of the sample: 192 frames, mean interval 40.01 ms, identical to `load_clip` + `LipReader.resample`;
  real-time pacing max lag 0.7 ms.
- Crop ms/frame at k = 2 (5090 box CPU, not the laptop): 4.75 full res 640x360, 5.00 at 1280x720; downscaled
  detection no faster (MediaPipe resizes internally), face ROI slower and less accurate. Laptop: estimate 8-15 ms.
- `laptop_check.sh` here: OK driver/CUDA 13.3, venv, mediapipe/cv2, landmarker, power readout, matmul, encoder;
  FAIL no webcam, not in `input` group (expected on this box); SKIP the rest.
- Camera at 60 fps resampled to 25 (every-other-frame gives 30 fps, not 25); only 50 fps keeps every other frame.

### B0 (bus + stub), 2026-10-24 ~04:45 UTC (estimate 45 min; took ~20 min)

- `venv-live`: `uv pip install pyzmq msgpack` -> pyzmq 27.2.0 (libzmq 4.3.5), msgpack 1.2.3, 0.6 s; torch/numpy untouched.
  `laptop_check.sh` section 4 now checks them (OK here).
- Vocabulary checked: model ids = `unigram5000_units.txt` line numbers (blank 0, 1..5047, `<eos>` 5048, odim 5049).
  SentencePiece's own ids differ (`▁WATER` 4899 vs spm 334): documented in `bus/schema.md` for Austin.
- `tests/lipread`: 56 passed in 22.1 s (49 old + 7 in `test_bus.py`).
- Stub over real ZMQ (`bus.stub` -> `bus.listen`, 30 s, this box): 111 cam.tick + 5 cam.final, 0 invalid,
  0 dropped (seq), payload max 53.5 KB (T = 200), bus latency (now - t_pub) p50 0.26 ms, p95 0.32 ms.
- Offline stub run, 150 s simulated: 531 ticks + 31 finals, all valid; max payload 54.6 KB.

### E1 finish (blendshapes + smoothing setting), 2026-10-24 (estimate 1 h)

- `lipread/crop.py`: `BlendTracker` (Lipflow FaceTracker options + `output_face_blendshapes=True`); `Crop` gains `blend`
  (52 float32, names in `BLEND_NAMES`), `mouth_pts` (13, 14, 61, 291) and `eye_dist`. Landmarks unchanged: k = 6
  symmetric still bit-exact vs `mouth_rois`. Smoothing presets `SMOOTHING` = k0 (past-only mean of last 6 + current),
  k2 (same + 2 frames lookahead), ema (alpha 0.3, k = 0). `tests/lipread`: 58 passed (2 new crop tests).
- Blendshapes cost (5090 box CPU, sample 20.4-28.1 s, median of 170 frames, 2 runs): detect 4.22/4.28 ms off vs
  4.28/4.28 ms on: free. Face + crop per frame (`crop_view`, 192 frames): k0 median 4.66 ms p95 5.73; k2 median
  4.73 p95 6.14. Laptop: estimate 8-15 ms (T1 pending).
- Smoothing vs the training-style centred crop (`lipread/bench/smooth_check.py`, whole sample video, 4711 frames, shared
  landmarks, 20 windows of 8 s with >= 90% faces, greedy fp32; "WER" = against the centred crop's greedy text, not truth):

  | Mode | Delay | px diff mean / p99 | WER vs centred | Same text |
  | --- | --- | --- | --- | --- |
  | box k2 | 80 ms | 2.04 / 7.9 | 0.054 | 8/20 |
  | box k0 (past only) | 0 | 2.95 / 10.0 | 0.054 | 7/20 |
  | ema 0.3 | 0 | 2.54 / 9.2 | 0.065 | 5/20 |
  | ema 0.5 | 0 | 2.73 / 9.6 | 0.068 | 6/20 |
  | ema 0.7 | 0 | 3.00 / 10.3 | 0.070 | 3/20 |

  Decision: default `k0` (past-only, as the team's workflow says, 0 delay, same disagreement as k2). EMA is worse
  than the plain past mean despite smaller pixel differences. Caveat: one speaker (newsreel, voiced, not silent
  mouthing), differences are a few words out of ~370; re-check on Austin's set E.
- Debug view `lipread/crop_view.py` (window, or MP4 when there is no display: WSL here has none). Mouth centred as in
  Lipflow's crops. On this speaker's resting face browDown ~0.55 and eyeSquint ~0.5: E5 needs the neutral baseline.

### Review fixes, 2026-10-24 ~05:35 UTC

- Own adversarial testing: `validate` accepted NaN log-probs/activity/confidence/scores, bool seg_id, duplicate ids
  in a row; a str seg_start raised TypeError. `decode` raised 5 exception types on garbage (bus_listen would crash).
  `encode` silently truncated float ids and wrapped int64 > int32. All fixed + tests.
- From the (stopped) review agent: `Publisher.close()` with LINGER 0 lost the last cam.final in 4/20 runs (measured)
  -> close waits up to 0.5 s; `bus_listen` counted a publisher restart as dropped=-40 -> counts restarts.
- Bus burst: a PUB socket drops past SNDHWM 20 in a burst (2 threads x 2000 msgs: 3110/4000 arrived, in order, no
  dups); at 5 Hz it doesn't. 500 sends of T=200 to a non-reading subscriber: 0.020 ms/send, never blocks.

### E2 (re-read loop), E4 (segments, subagent), E7 recorder (subagent), 2026-10-24 ~06:00 UTC

- `lipread/model.py` LiveModel: encoder on the whole segment, CTC log_softmax + top-32 + blank on GPU, one copy to host.
  Greedy text identical to `LipReader.greedy` on the sample; top-1 == CTC argmax. Warm-up of every length 1..200
  (4.3 s): the first read at a new length was up to 62 ms (5090), then 11-14 ms.
- `lipread/run.py`: crop thread -> queue -> tick loop; a tick per 5 new frames (deterministic in fast replay; live, waits
  for the queue to drain up to 10 frames); last read of a segment at cap / end; replay times rebased to monotonic;
  repeated frames get the next 40 ms slot so `frame_t` strictly increases. `--segments auto|stream`.
- Sample 20.4-28.1 s, fast replay (5090, agents running): 38 ticks, tick total p50 18.5 ms p95 23.3; peak GPU
  1310 MiB; crop 5.2 ms/frame; listener: 38/38 valid, bus latency p50 0.32 ms. Greedy settles on the sentence.
- Real-time replay 15-37 s with E4 segments: 4 segments (3 caps at 8.00 s with 50 frames carried, 1 end), 112
  ticks, 112/112 received valid. Tick total p50 15.5 ms, **p95 51-52 ms** (2 runs), lag (capture -> published)
  p50 22 ms p95 59 ms. The slow ticks are runs of consecutive ticks whose time grows with T (to ~60 ms at T = 200).
  Standalone paced reads (200 ms apart) reproduced it once (p95 54 ms) and not the next time (p95 24 ms); back to
  back always ~13 ms; a keep-warm GPU thread made it worse (p95 51). Something else holds 1.1 GB on this GPU and the
  GPU idles at P5 / 240 MHz between ticks: dev-box timings are too noisy to conclude. **Decide on the laptop (E3).**
- E4 (`lipread/endpoint.py`, subagent): Activity = lip gap / width / corner-lift ranges over 300 ms, eye_dist-normalised,
  3-frame median; Segmenter start 0.25 / still 0.12, confirmation 0.4 s (start event ~0.36 s after the rise, pre-roll
  kept), 0.5 s still ends, cap 200 / carry 50. Sample: one segment covering 7.68/7.68 s; synthetic still / nod /
  jitter max activity 0.075 / 0.088 / 0.078. Chewing and real blinks untested (no clips).
- E7 (`lipread/record.py`, subagent): H.264 crf 18 veryfast via PyAV, VFR pts = capture times, ts 1:1 with frames
  (drops absent from both). 640x480: 4.2 MB/min, 8.4 CPU ms/frame, crop diff vs original 2.17 (lossless floor 1.48,
  mostly cv2 decode darkening ~1.25 grey levels). EMG merge via `--manifest manifest_video.jsonl` + `--merge`.
- `tests/lipread`: 86 passed (66.2 s).

### E5 handed to Frank; E3 prep, 2026-10-24 ~06:30 UTC

- E5: Evan gave emotion to Frank's session (calhacks2026-39), which owns lipread/emotion*.py and the crop/run hooks.
  Our blendshape EmotionTagger (subagent, 18 tests) was overwritten by accident there at 06:14; recovered from the
  subagent transcript into lipread/emotion.py, now Frank's to keep or replace.
- fp16 parity (`.half()` vs fp32, sample video, 51 windows of 2/4/8 s): greedy text identical 51/51, top-1 token
  agreement 99.98% of frames, mean |delta logp| 0.0003. Gate passed: fp16 allowed.
- `lipread/bench/tick_bench.py` (the laptop E3 command): segments growing 5 frames per tick, 200 ms apart. 5090, with
  the E6 agent benchmarking on the same GPU (noisy): fp32 p95 20.1 / 32.2 / 32.3 ms for 2 / 4 / 8 s, peak 1310 MiB;
  fp16 p95 38.5 / 36.7 / 29.7 ms, peak 966 MiB, warm-up 26.4 s (vs 5.2 s fp32). Laptop numbers decide.

### E6 final pass (subagent), 2026-10-24 ~07:00 UTC

- `lipread/final.py` FinalPass: beam 5 + LM, nbest 5, in a spawned subprocess that runs only the decoder + LM on CPU
  (4 threads, nice 10) on the tick's encoder output (`LiveModel.read(keep_enc=True)`, new optional `Reading.enc`).
  Queue: 1 running + 2 waiting, drop-oldest (counted, logged). Process start ~25-35 s (build during warm-up).
- Tick cost of a concurrent 8 s beam (5090, shared/noisy GPU, ticks every 40 ms on T 5..200; tick p50 / p95 ms):
  baseline 13-16 / 20-31; thread on GPU 52 / 69 (GIL); thread + switchinterval or own CUDA stream no better;
  subprocess with GPU decoder 23 / 50 (+1.16 GiB, second CUDA context); subprocess CPU nice 0 22 / 53;
  **subprocess CPU nice 10: 17.7 / 31.9 (baseline that run 13.4 / 24.4), 0 extra VRAM. Chosen.**
- Beam time, CPU 4 threads / GPU: 2 s 490 / 350 ms, 4 s 800 / 650 ms, 8 s 1500 / 1350 ms. Sample through the
  process path: best = all 16 truth words ("... AND RAISED ... IN NINETEEN FORTY THREE"); 1505 ms.
- Scores are raw weighted log-prob sums (0.9 decoder + 0.1 CTC + 0.3 LM), not length-normalised: tell Austin.
  `tests/lipread/test_final.py` 8 passed. Laptop CPU unknown: re-measure beam ms and tick p95 there.

### Next block (3 subagents + lead), 2026-10-24 ~07:00 UTC

- Team docs changed (Austin, 05:39 / 06:06): cam.tick every 100 ms, ipc://, `timing_ms`; E7 needs replay only.
- Bus v2 (`bus/__init__.py`, v = 2): default `ipc:///tmp/reclaim/cam.tick`, required `timing_ms` {T, mediapipe, frontend,
  conformer, ctc, total}, stale socket file handled, close under the send lock. Stub + listener 10 s over ipc: 63 ticks
  + 2 finals, 0 invalid, 0 dropped, latency p50 0.36 ms.
- run.py: 100 ms ticks (due points 3, 5, 8, 10, ...), crop thread owns source + cropper, stall flush, non-zero exit on
  thread death, per-tick try/except, Ctrl+C last read, backlog coalescing, frame times assigned once, `<unk>` dropped.
- source/endpoint/crop_view/record: camera stall reopen + `CameraSource.error`, no shared pre-roll after an end,
  crop_view FIFO pairing, record temp-then-rename; ReplaySource reads per-frame timestamps (team RECORDING.md layout
  and svr/recordings.py layout; the team recorder code doesn't exist yet).
- access-35 session (code-quality pass, 06:32): Segmenter.stream() replaces StreamSegmenter; TickStat. Kept.
- E6 wired into run.py (lead): last read's encoder output reused; beam process started during warm-up; `--no-final`.
- Real-time replay 15-37 s, 100 ms ticks, emotion (Frank's EmotionWorker) + final pass on (5090, shared GPU): 219
  ticks + 4 cam.final, all valid, 0 dropped; tick total p50 18.1 ms **p95 38.6** max 84.5; lag p50 26 ms p95 55;
  4 beams, 0 dropped; peak GPU 1311 MiB. Finals fix greedy look-alikes ("AND RAISED", "NANCY REAGAN").

### E3 wrap-up (lead), 2026-10-24 ~07:40 UTC (5090 dev box; laptop numbers still owed)

- Front-end cache (E3 step 2): `model.FrontendCache` + `LiveModel.encode_cached`; the Conv3d front end sees +-2
  frames, so a frame's features are final once 2 later frames exist; the newest 2 are recomputed with the same zero
  padding; new cache per segment (also after a cap). Parity on the sample, every 100 ms tick: fp32 greedy identical
  80/80 and in the Runner test (with a cap); fp16 77/80; top-1 logp differs up to 0.05 (TF32 conv algorithms).
  Speed on the 5090: front end is only ~2-4 ms at 8 s (Conformer ~9-10 ms dominates); cache vs none within noise
  (8 s p95 27.6 vs 39.5 ms in one paired run, other runs reversed). On by default in run.py, `--no-fe-cache` off.
- `cudnn.benchmark`: measured in separate processes, idle GPU: fp32 warm-up 3.8 -> 80.4 s, peak 1.3 -> 20.0 GB,
  paced p95 17.2 -> 18.4 ms; fp16 warm-up 19.5 -> 80.2 s, p95 22.9 -> 25.1 ms. No gain, would break the 8 GB laptop:
  now `LiveModel(cudnn_benchmark=False)` by default (the team doc's step 1 says True: evidence against it).
- fp16 (`--half`): parity OK (51/51) but not faster on the 5090 (paced p95 22.9 vs 17.2 ms fp32) and 19.5 s warm-up.
  fp32 stays the default; decide on the 5060.
- E3 status: steps 1 (model once, warm-up all lengths; fp16 optional), 2 (cache), 3 (8 s cap, 2 s carry), 4
  (MediaPipe in its own thread) and timing_ms done. Target p95 < 40 ms at 8 s met on the 5090 in every run
  (17.5-39.5 ms). Deliverable still owed: p50/p95 for 2/4/8 s on Austin's laptop:
  `python -m lipread.bench.tick_bench` (it times fp32 and fp16; and `--no-fe-cache` once).

### fp16 by default, 2026-10-24

- `lipread.run` now runs the encoder in fp16 (`.half()`) by default, `--fp32` for the old path: the team chose fp16
  for the 8 GB laptop (peak 966 vs 1310 MiB in `tick_bench` above) and its parity is checked (51/51 greedy
  identical). Costs on the 5090: paced p95 22.9 vs 17.2 ms and a longer warm-up (19.5 vs 3.8 s).
