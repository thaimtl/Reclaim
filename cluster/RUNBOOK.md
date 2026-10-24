# Cluster runbook (night of day 1)

Give this file to the Claude session on the cluster (or follow it by hand). Read `CONTEXT.md` first.
Clusters use PBS. Prefer Moira (H200) if a node is free, else Conrad (H100). One job per GPU.

## 0. Login and pull everything (once)

```bash
hf auth login                      # token with access to the private bucket austinkomachi/Austin_CH
mkdir -p $WORK && cd $WORK         # $WORK = a scratch/project directory with ~40 GB free
hf buckets cp hf://buckets/austinkomachi/Austin_CH/repo/cluster/setup.sh ./setup.sh
bash setup.sh $WORK/calhacks2026   # code + data + Gaddy checkpoints + both venvs + camera weights
cd $WORK/calhacks2026
```

If a step of `setup.sh` fails (package build, Google Drive quota, etc.), fix it and rerun; it is idempotent.

## 1. Smoke-test the EMG baselines (interactive GPU session, ~10 min each)

```bash
bash cluster/smoke.sh G6_baseline --keep-channels 0,1,2,3,4,5 --micro-batch-samples 128000 --grad-accum 2
bash cluster/smoke.sh B1T_exact --micro-batch-samples 128000 --grad-accum 2
```

Each must print `SMOKE PASS`. It reports peak VRAM (worst micro-batch of the whole run) and peak RAM.
Fix any failure in the code, rerun the smoke test, and push the fix (step 6).

## 2. Set PBS resources

Edit `cluster/train_job.sh`: queue/account for the cluster, `ngpus=1`, `mem` = 1.5x the smoke-test peak RAM,
`ncpus=4`, `walltime=12:00:00`. These baselines use Gaddy's batch size on purpose (~50k optimizer steps); do not
raise the micro-batch for them.

## 3. Launch the baselines

Always submit through `cluster/submit.sh` (never `qsub -v ARGS=...` directly: qsub splits `-v` values on commas,
which breaks `--keep-channels 0,1,2,3,4,5` and `8ch,6ch`). Queue/account flags go in `QSUB_OPTS`:

```bash
export QSUB_OPTS="-q <queue> -A <account>"     # whatever the cluster needs; reused by self-resubmits
bash cluster/submit.sh G6_baseline 6ch --keep-channels 0,1,2,3,4,5 --micro-batch-samples 128000 --grad-accum 2
bash cluster/submit.sh B1T_exact 8ch,6ch --micro-batch-samples 128000 --grad-accum 2
```

Plus one short evaluation-only job (Gaddy's released model fed our 6 channels):

```bash
bash cluster/baselines.sh B1_6ch
```

Each training run takes roughly 1-3 h, checkpoints every 15 min, resubmits itself at the wall time, restarts
itself after a crash (supervisor), and benchmarks itself when done.

## 4. Smoke-test the camera chain (no recordings needed)

```bash
bash third_party/lipflow/scripts/download-models.sh --samples
venv-camera/bin/python -m camera.smoke third_party/lipflow/samples/2016-03-12.mov
```

Must print `CAMERA SMOKE PASS`. Likely fixes: BRAVEn LM file format from Google Drive, LipLearner checkpoint key
names, raven's old espnet on Python 3.11, the Qwen model id (`--llm`).

## 5. Monitor (every ~30-45 min)

```bash
venv/bin/python -m svr.status G6_baseline
venv/bin/python -m svr.status B1T_exact
qstat -u $USER
```

Report: epoch, train loss, dev WER (greedy and beam), GPU-hours, TFLOPS, VRAM, GPU util, CPU/RAM (Erbium).
If a job crashed beyond the supervisor's restarts: read `checkpoints/<run>/train_stdout.log`, fix the code, rerun
the relevant smoke test, resubmit with the same `cluster/submit.sh` command (it resumes from the last checkpoint).

Expected: B1T_exact should land near Gaddy's 27% WER (test) / 28% (dev). G6_baseline is the 6-channel number SVR-1
must beat.

## 6. Push results and fixes to the bucket

```bash
B=hf://buckets/austinkomachi/Austin_CH
hf sync ./benchmark_results $B/results/benchmark_results
for r in G6_baseline B1T_exact; do hf sync ./checkpoints/$r $B/checkpoints/$r --exclude "ckpt/*"; done
for d in svr bench tests camera cluster; do hf sync ./$d $B/repo/$d --exclude "*__pycache__*"; done
```

Never use `--delete`.

## 7. SVR-1

Code is in the bucket. `hf sync $B/repo .` and `hf sync $B/data ./data` (new: `data/lexicon/g2p_supplement.dict`),
then follow `CONTEXT.md` section "SVR-1 status": SPARC targets -> `bash cluster/smoke_svr1.sh SVR1_full` ->
`bash cluster/svr1_sweep.sh SVR1_full`, then the rest in priority order (`bash cluster/svr1_sweep.sh list`).
Monitor with `venv/bin/python -m svr.status SVR1_full`. If the SPARC step blocks, start `SVR1_full` with
`--no-artic` rather than waiting (and say so in the results).
