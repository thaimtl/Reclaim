# Cluster runbook: SVR-1 pretraining

For the Claude session on the cluster (or a person). Goal: **SVR-1 pretraining on Berkeley's public data finished, and its weights, logs and benchmarks in the bucket.** Austin may be asleep, so you work alone until that is true. The design is in [docs/SVR1_SPEC.md](../docs/SVR1_SPEC.md).

Two runs, one GPU each:

| Run | What | Priority |
| --- | --- | --- |
| `SVR1_4ch` | The main run: SVR-1 on the rig's 4 sites, without articulator targets | Must finish |
| `SVR1_4ch_artic` | The same with SPARC articulator targets; starts when they are ready | Best effort, never at the main run's expense |

**Out of scope tonight:** the baselines (`G4`, `G6_baseline`, `B1T_exact`, `cluster/baselines.sh`), the camera models, anything that downloads from Google Drive (a 404 there is known; ignore it), multi-GPU training, and fine-tuning on Austin's recordings.

## Standing orders

1. Austin's message gives you the project root on this cluster. **Everything you create stays inside it:** the working copy, environments, caches, temp files, job logs. `cluster/env.sh` (sourced by every script) points every cache, temp folder and the Hugging Face token into the working copy. Source it in your own shells too: `source cluster/env.sh`.
2. Keep going until the main run is finished and synced (end condition below). Check every 20 to 30 minutes. Don't stop to ask; nobody will answer before morning.
3. When something fails, read the log, find the cause, fix it, rerun the step's check, and resume. "If something fails" below has the usual cases. Runs resume from their last checkpoint; never delete `checkpoints/<run>/`.
4. **Every job you submit is named `Austin-Train`**, nothing else (Austin's rule). The scripts already do this (`-N Austin-Train`); add `-N Austin-Train` to any `qsub` you type yourself. Tell the jobs apart by their output files (`cluster/jobs/<run>.pbs.out`, `checkpoints/_prepare/pbs.out`, `checkpoints/_sparc/pbs.out`) and `qstat -f <id>` (Variable_List shows the run).
5. Code fixes: edit in the working copy so the run can continue, and list every changed file in the report. Copy the changed files to the bucket under `cluster_fixes/<date>/`, with the same relative paths (`hf buckets cp <file> hf://buckets/austinkomachi/Austin_CH/cluster_fixes/<date>/<path>`). Austin merges them into the repo in the morning. Don't overwrite the bucket's `repo/`.
6. The bucket: only `hf sync` without `--delete`. Never remove anything there. Never sync the bucket's `calhacks2026/` folder into the working copy: it is a full snapshot of Austin's machine, including Python environments that don't run here and would replace yours.
7. Keep `checkpoints/NIGHT_REPORT.md` up to date: what you started and when, every failure and fix, the latest epoch, dev WER, ETA, and where the results are.

## 1. The working copy (code and data from the bucket, no git)

The code comes from the bucket's `repo/` folder: a snapshot of the Silent Voice repo, this runbook included, since the cluster can't reach GitHub. It is uploaded only with `cluster/upload_snapshot.sh`, from a clean checkout of a pushed commit. `cluster/VERSION` holds that commit (`git rev-parse HEAD`), `cluster/setup.sh` prints it, and the night report records it. Data comes from `data/`.

```bash
cd <project root>
ls          # find the working copy: the folder with silent_speech/, data/, checkpoints/ and venv-svr/
cd <working copy>
hf sync hf://buckets/austinkomachi/Austin_CH/repo .     # the code (only adds and updates)
bash cluster/setup.sh                                  # data, Gaddy checkpoints, venv-svr if missing
source cluster/env.sh
```

If there is no working copy yet, make the folder `<project root>/calhacks2026` and run the same commands in it.

The bucket's `data/` already holds the causal cache (`data/preprocessed_causal/`) and `data/svr1_targets/scale.json`, built on Austin's machine, so the preparation job's cache step only checks it. It finishes in seconds.

`setup.sh` skips the camera environment; `WITH_CAMERA=1` would build it. Don't tonight.

Python: use `venv-svr/bin/python`. It needs `numba` and `scipy`; check with `venv-svr/bin/python -c "import torch, numba, scipy; print(torch.__version__, torch.cuda.is_available())"`. If `venv-svr` is missing, `cluster/setup.sh` builds it from `requirements-cluster.txt`.

Hugging Face login: `env.sh` keeps the token in `.cache/huggingface/` inside the working copy, copying an existing login from `~/.cache/huggingface/token` the first time. `hf auth whoami` must work after `source cluster/env.sh`. If it doesn't, start anyway (`pretrain.sh` only warns), note it in the report, and have Austin log in in the morning; the sync catches up then.

## 2. Start everything (one command)

```bash
export QSUB_OPTS="-q <queue> [-A <account>]"      # how this cluster takes GPU jobs; prefer an H200 queue if free
export FLASH_ATTN_WHEEL=<wheel path from Austin's message>
bash cluster/pretrain.sh
```

`QSUB_OPTS` holds only the queue and account; the job scripts set their own resources (`select=1:ncpus=8:mem=32gb:ngpus=1` for training, 16 CPUs and 64 GB for the preparation jobs). If you don't know the queue, look at `qstat -Q` and at the `.qsub` files of earlier runs in `cluster/jobs/`. If this cluster rejects `vntype=gpu`, edit the `#PBS -l select` lines (in `train_job.sh`, `prepare_job.sh`, `sparc_job.sh`) and note it for the fixes branch.

What `pretrain.sh` does:

1. **On the login node:** checks the data and the Hugging Face login, installs FlashAttention-2 into `venv-svr` from the wheel, then builds `venv-sparc` and downloads the SPARC weights.
2. **Preparation job** (`cluster/prepare_job.sh`, 1 GPU, about 30 to 45 minutes):
    - Builds the causal cache: Berkeley re-filtered with `rig/filters.py`, plus the scale statistics.
    - Reports which attention kernel will run.
    - Runs the smoke test: unit checks, a tiny training run with a simulated crash and resume, the benchmark chain, and peak VRAM.
    - Submits `SVR1_4ch`.
    - Log: `checkpoints/_prepare/prepare.log`, ending in `PREPARE PASS` or `PREPARE FAIL: <why>`.
3. **SPARC job** (`cluster/sparc_job.sh`, 1 GPU, on the side):
    - Makes the articulator targets, waits for the cache, aligns the targets to it, runs the smoke test, then submits `SVR1_4ch_artic`.
    - Log: `checkpoints/_sparc/sparc.log`, ending in `SPARC PASS` or `SPARC FAIL`.
    - On failure it writes `checkpoints/_skipped_SVR1_4ch_artic`.
4. **Bucket watcher** (tmux session `reclaim_sync`, log `checkpoints/_sync_watch.log`): every 30 minutes, and once more when each run ends, it syncs `checkpoints/<run>/` (not `ckpt/`), `benchmark_results/` and `data/svr1_targets/` to `hf://buckets/austinkomachi/Austin_CH`. Each training job also syncs once when it finishes, in case the watcher dies.

## 3. Training, and what to expect

Each training run lasts 150 epochs (about 61,000 steps). Estimate: about 1 to 1.5 minutes per epoch, so 2.5 to 4 hours; the real ETA appears after the first epoch.

```bash
venv-svr/bin/python -m svr.status SVR1_4ch        # epoch, losses, dev WER, best epoch, ETA (on a login node: no GPU line)
qstat -u $USER
tail -3 checkpoints/SVR1_4ch/log.jsonl
```

- `model.pt` is the best epoch so far by dev WER (greedy); `model_last.pt` is the latest.
- **Early stopping:** the run ends when dev WER hasn't improved for 20 epochs, but never before epoch 60. `DONE` records whether it stopped early.
- `train_speak` and `train_cut_rows` appear from epoch 21, after the first forced alignment. Check `train_stdout.log` for `epoch 21: forced alignments for N/M utterances`: N should be close to M. `train_cut_rows` should be about 0.3.
- Dev WER is greedy, with no language model, at 4 sites. It starts at 100% and falls over the first tens of epochs. 4 sites is much harder than 8, so don't compare it with the old 6- or 8-channel numbers.
- Walltime is 12 hours. A job that hits it checkpoints and resubmits itself; `svr/supervise.sh` restarts crashes up to 5 times. When training ends, it benchmarks the model at 4 sites (`benchmark_results/SVR1_4ch_ours_{gpu,cpu}/`).

## 4. If something fails

| Symptom | What to do |
| --- | --- |
| `PREPARE FAIL: cache` | Read the traceback in `prepare.log`. Usually a missing package (install it into `venv-svr`) or missing data (`silent_speech/emg_data`: rerun `setup.sh`). Then `qsub $QSUB_OPTS -N Austin-Train -o $PWD/checkpoints/_prepare/pbs.out -v RUN=SVR1_4ch cluster/prepare_job.sh`. |
| `PREPARE FAIL: smoke test` | The failing check is printed just above. Fix the code, then reproduce on a GPU with `bash cluster/smoke_svr1.sh SVR1_4ch --no-artic`, or resubmit `prepare_job.sh`. The CPU unit checks also run on a login node: `CUDA_VISIBLE_DEVICES= venv-svr/bin/python -m tests.smoke_svr1_units`. |
| Out of GPU memory in the smoke test's worst-batch step | Lower the micro-batch: add `--micro-batch-samples 80000` to `MAIN_ARGS` (`MAIN_ARGS="--no-artic --micro-batch-samples 80000"`) and resubmit the preparation job. Note it in the report. |
| A job is killed for memory (RAM) | Raise `mem` in the job script's `#PBS -l select` line (training: 32gb, so try 64gb) and resubmit. |
| Training crashed and the supervisor gave up (`checkpoints/<run>/supervisor.log`) | Read `checkpoints/<run>/train_stdout.log`, fix, then resume with `bash cluster/submit.sh --svr1 SVR1_4ch 4ch --no-artic`, the same arguments. It continues from `ckpt/latest.pt`. "refusing to resume: config differs" means the arguments changed: use the original ones (`cluster/jobs/SVR1_4ch.args`). |
| The job sits in the queue | Fine. If a different queue or GPU type is free, you may resubmit there with the same arguments. Kill the waiting job first (`qdel`). |
| `SPARC FAIL` | Best effort: try one fix (install, weights, a shape change in `svr/precompute_svr1.py stage_sparc`), then `rm checkpoints/_skipped_SVR1_4ch_artic` and `qsub $QSUB_OPTS -N Austin-Train -o $PWD/checkpoints/_sparc/pbs.out -v RUN=SVR1_4ch_artic cluster/sparc_job.sh`. If it still fails, leave it and say so in the report. |
| flash-attn won't import | It still trains, with PyTorch SDPA (same results, a bit slower). Don't build flash-attn from source tonight. |
| `hf sync` fails | Training continues. Retry on the login node: `bash cluster/sync_results.sh SVR1_4ch SVR1_4ch_artic`. |
| Watcher dead | `tmux new-session -d -s reclaim_sync "bash cluster/sync_results.sh --watch SVR1_4ch SVR1_4ch_artic >> checkpoints/_sync_watch.log 2>&1"` |

## 5. End condition

Done when all of these hold:
- `checkpoints/SVR1_4ch/DONE` exists.
- `checkpoints/SVR1_4ch/supervisor.log` says `benchmarks done`.
- A final `bash cluster/sync_results.sh SVR1_4ch` succeeded.
- `hf buckets ls hf://buckets/austinkomachi/Austin_CH/checkpoints/SVR1_4ch` lists `model.pt`.

Do the same for `SVR1_4ch_artic` if it is running, but don't wait past the main run's end to start fixing it.

Finish `checkpoints/NIGHT_REPORT.md` with:
- the code version (`cluster/VERSION`);
- the best epoch and its dev WER and homophene-10;
- whether the run stopped early;
- GPU-hours and the attention kernel used;
- the benchmark summary (`benchmark_results/SVR1_4ch_ours_gpu/summary.json`);
- every fix you made, and where its files are in the bucket (`cluster_fixes/<date>/`).

Sync the report too: `hf buckets cp checkpoints/NIGHT_REPORT.md hf://buckets/austinkomachi/Austin_CH/checkpoints/NIGHT_REPORT.md`.

## Later (not tonight)

- Fine-tuning on Austin's sessions, on any one GPU: `venv-svr/bin/python -m svr.finetune --name W_svr1 --init-from checkpoints/SVR1_4ch/model.pt`. It needs `recordings/session_0N/` from the bucket.
- The baselines with Gaddy's original inputs: `cluster/baselines.sh` (they pass `--preprocess gaddy`).
- `G4`: `bash cluster/svr1_sweep.sh G4`.
