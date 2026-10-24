# Cluster runbook: SVR-1 pretraining

Give this file to the Claude session on the cluster, or follow it by hand. The design is in [docs/SVR1_SPEC.md](../docs/SVR1_SPEC.md): SVR-1 on Berkeley's data, re-filtered with `rig/filters.py`, on the rig's 4 sites (input slots 0, 1, 2, 5). This runbook covers stage 0 (inputs and targets) and stage 1 (pretraining `SVR1_4ch` and the baseline `G4`).

Clusters use PBS. Prefer an H200 node if one is free, else an H100. One job per GPU.

## 0. Set up or update the working copy (once)

The working copy is a git checkout of the Reclaim repo with the data in git-ignored folders beside the code.

```bash
hf auth login                      # token with access to the private bucket austinkomachi/Austin_CH
cd $WORK                           # a scratch/project directory with ~60 GB free
curl -fsSLO https://raw.githubusercontent.com/thaimtl/Reclaim/main/cluster/setup.sh
bash setup.sh $WORK/calhacks2026 main
cd $WORK/calhacks2026
```

`setup.sh` is idempotent. On a copy set up before (synced from the bucket, no `.git`), it turns the copy into a checkout of the branch, replacing the old code files and leaving the data folders alone. If a step fails (a package build, a quota), fix it and rerun.

The scripts use `venv-svr/bin/python`, or `venv/bin/python` on copies set up before the rename. Below, `PY` is whichever exists:

```bash
PY=venv-svr/bin/python; [ -x $PY ] || PY=venv/bin/python
```

## 1. Inputs and targets (stage 0, CPU, about 30 minutes)

Run on an interactive node with 16+ cores, before any training job:

```bash
$PY -m svr.precompute_svr1 cache     # rig/filters.py over every Berkeley recording -> data/preprocessed_causal/
                                     # + data/svr1_targets/scale.json (per-session and target spreads)
```

It prints the 8 per-slot target spreads (about 10 to 30). Then the articulator targets:

- If `data/svr1_targets/sparc/` already exists (pulled from the bucket), skip to `align`.
- Otherwise, run SPARC on a GPU, in any environment with `pip install speech-articulatory-coding soundfile`:

```bash
python -m svr.precompute_svr1 sparc --device cuda
```

Then, always, because it reads the new cache:

```bash
$PY -m svr.precompute_svr1 align     # -> data/svr1_targets/artic/ + stats.json
```

If SPARC blocks, don't wait: start `SVR1_4ch` with `--no-artic`, and say so in the results.

## 2. Smoke tests (interactive GPU session, about 15 minutes each)

```bash
bash cluster/smoke_svr1.sh SVR1_4ch
bash cluster/smoke.sh G4 --keep-channels 0,1,2,5 --micro-batch-samples 128000 --grad-accum 2
```

Each must print `SMOKE PASS`. The SVR-1 smoke test runs the unit checks, a tiny run with forced alignments, partial-sentence cuts, a simulated crash and resume, the benchmark chain at 4 slots, and the peak VRAM of the worst micro-batches of the full run. Fix any failure in the code, rerun, and push the fix as a PR (step 6).

## 3. PBS resources

`cluster/train_job.sh` holds the defaults. Pass the queue, account and memory through `QSUB_OPTS`, with `mem` set to 1.5× the smoke test's peak RAM, for example:

```bash
export QSUB_OPTS="-q <queue> -A <account> -l select=1:ncpus=8:mem=<n>gb:ngpus=1:vntype=gpu"
```

Self-resubmits reuse these options.

## 4. Launch

Always submit through `cluster/svr1_sweep.sh`, which calls `cluster/submit.sh`. Never call `qsub -v ARGS=...` directly: qsub splits `-v` values on commas, which breaks `--keep-channels 0,1,2,5`.

```bash
bash cluster/svr1_sweep.sh list        # the commands
bash cluster/svr1_sweep.sh SVR1_4ch    # main run: 150 epochs, ~61k steps
bash cluster/svr1_sweep.sh G4          # Gaddy's architecture and recipe, same 4 slots and inputs
```

Then, if GPUs are free, the optional runs in order: `SVR1_8ch_pre` (8 slots with channel dropout, then `SVR1_8to4`, see the script), and the ablations `SVR1_no_judge`, `SVR1_no_prefix`, `SVR1_no_delivery`, `SVR1_no_artic`.

Each run checkpoints every 15 minutes, resubmits itself at the wall time, restarts after a crash (`svr/supervise.sh`), and benchmarks itself at 4 slots when done.

## 5. Monitor (every 30 to 45 minutes)

```bash
$PY -m svr.status SVR1_4ch
$PY -m svr.status G4
qstat -u $USER
tail -3 checkpoints/SVR1_4ch/log.jsonl
```

Report: epoch, losses (`train_letter`, `train_phone`, `train_artic`, `train_rate`, `train_speak`, `train_judge`, `train_cut_rows`), dev WER (greedy), dev homophene-10 top-1, GPU-hours, VRAM, GPU use, CPU and RAM.

What to expect:

- `train_speak` and `train_cut_rows` appear from epoch 21, after the first forced alignment. The line `epoch 21: forced alignments for N/M utterances` should cover nearly every utterance.
- `train_cut_rows` should be about 0.3.
- Dev WER is greedy, without a language model, at 4 slots. Gaddy and Klein's 4-site result is much worse than 8 sites, so compare `SVR1_4ch` with `G4`, not with the old 6- or 8-channel numbers.

If a job crashes beyond the supervisor's restarts, read `checkpoints/<run>/train_stdout.log`, fix the code, rerun the smoke test, and resubmit with the same command. It resumes from the last checkpoint.

## 6. Push results and fixes

Results and checkpoints go to the bucket. Never use `--delete`.

```bash
B=hf://buckets/austinkomachi/Austin_CH
hf sync ./benchmark_results $B/results/benchmark_results
for r in SVR1_4ch G4; do hf sync ./checkpoints/$r $B/checkpoints/$r --exclude "ckpt/*"; done
hf sync ./data/svr1_targets $B/data/svr1_targets
```

Code fixes go to the repo as a PR: commit on a new branch, push it, and open the PR. Main is protected.

## 7. After pretraining

`checkpoints/SVR1_4ch/model.pt` is the start of fine-tuning on Austin's sessions 1 and 2. It carries `preprocess: causal` and Berkeley's `scale_target`, so fine-tuning and the live loop scale Austin's signal the same way:

```bash
$PY -m svr.finetune --name W_svr1 --init-from checkpoints/SVR1_4ch/model.pt      # sessions 1-2 -> test on 3
$PY -m svr.finetune --name W_g4 --arch gaddy --init-from checkpoints/G4/model.pt  # the baseline, same data
```

These need the recordings in `recordings/session_0N/` (from Austin's laptop) and run on any single GPU.
