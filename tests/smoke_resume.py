"""Smoke-test checkpoint saving and resuming of svr.train through the real CLI.

Scenarios (all tiny: a few micro-batches per epoch, 3 epochs):
  1. hard crash mid-epoch (os._exit at optimizer step 8, checkpoint every step) -> resume
  2. SIGTERM mid-run -> exit code 3 + checkpoint -> resume
  3. gradient accumulation 2 with an odd number of micro-batches per epoch: crash after an
     epoch-end checkpoint that holds half-accumulated gradients -> resume
  4. truncated latest.pt -> resume falls back to prev.pt
  5. resuming with a different config is refused and leaves the checkpoint untouched
Each resumed run is compared with an uninterrupted reference, micro-batch by micro-batch:
same recordings, shift, learning rate, and losses within GPU-nondeterminism tolerance.

    venv/bin/python -m tests.smoke_resume
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
OUT = os.path.join(WORK, 'checkpoints', '_smoke')
PY = next((p for p in (os.path.join(WORK, v, 'bin', 'python') for v in ('venv-svr', 'venv')) if os.path.exists(p)),
          sys.executable)
BASE = ['--micro-batch-samples', '432000', '--epochs', '3', '--eval-limit', '3', '--beam-every', '1000',
        '--monitor-seconds', '5', '--step-log', '--out-root', OUT]
LOSS_RTOL = 2e-3
failures = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def train(name, *extra, wait=True):
    cmd = [PY, '-m', 'svr.train', '--name', name] + BASE + list(extra)
    log = open(os.path.join(OUT, name + '.stdout'), 'a')
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return p.wait() if wait else p


def steps(name):
    """micro_idx -> last logged record (a resumed run re-logs the micro-batch it redoes)."""
    out, first = {}, {}
    with open(os.path.join(OUT, name, 'steps.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            first.setdefault(r['micro_idx'], r)
            out[r['micro_idx']] = r
    return out, first


def compare(ref_name, name, label):
    ref, _ = steps(ref_name)
    got, _ = steps(name)
    check(f'{label}: same number of micro-batches', sorted(ref) == sorted(got), f'{len(got)} vs {len(ref)}')
    same_data = all(ref[k]['first_id'] == got[k]['first_id'] and ref[k]['n_recordings'] == got[k]['n_recordings']
                    and ref[k]['shift'] == got[k]['shift'] and ref[k]['lr'] == got[k]['lr'] for k in ref if k in got)
    check(f'{label}: identical batches, shifts and learning rates', same_data)
    worst = max(abs(ref[k]['loss'] - got[k]['loss']) / abs(ref[k]['loss']) for k in ref if k in got)
    check(f'{label}: losses match the uninterrupted run', worst < LOSS_RTOL, f'max relative diff {worst:.2e}')
    done = os.path.exists(os.path.join(OUT, name, 'DONE'))
    check(f'{label}: run completed (DONE written)', done)


def stdout(name):
    with open(os.path.join(OUT, name + '.stdout')) as f:
        return f.read()


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    t0 = time.time()

    # references
    check('reference run (defaults: 432k, accum 1) exits 0', train('ref1', '--max-micro-batches-per-epoch', '6',
                                                   '--ckpt-minutes', '0') == 0)
    check('reference run (accum 2, odd epoch length) exits 0',
          train('ref2', '--max-micro-batches-per-epoch', '5', '--grad-accum', '2', '--ckpt-minutes', '999') == 0)

    # 1. hard crash mid-epoch
    code = train('crash', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0', '--crash-at-step', '8')
    check('1 crash: process died with exit code 1', code == 1, f'exit {code}')
    code = train('crash', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0')
    check('1 crash: resumed run exits 0', code == 0)
    check('1 crash: resumed from step 7 (last checkpoint before the crash)',
          'optimizer step 7' in stdout('crash'), [l for l in stdout('crash').splitlines() if 'resumed' in l][:1])
    _, first = steps('crash')
    last, _ = steps('crash')
    redo = abs(first[8]['loss'] - last[8]['loss']) / first[8]['loss']
    check('1 crash: redone step 8 reproduces the crashed run\'s loss', redo < 1e-4, f'relative diff {redo:.2e}')
    compare('ref1', 'crash', '1 crash')

    # 2. SIGTERM mid-run
    p = train('sigterm', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0', wait=False)
    path = os.path.join(OUT, 'sigterm', 'steps.jsonl')
    while p.poll() is None:
        if os.path.exists(path) and sum(1 for _ in open(path)) >= 9:
            p.send_signal(signal.SIGTERM)
            break
        time.sleep(0.2)
    code = p.wait()
    check('2 SIGTERM: exit code 3 (interrupted, resumable)', code == 3, f'exit {code}')
    check('2 SIGTERM: checkpoint saved before exiting', 'checkpoint saved at optimizer step' in stdout('sigterm'))
    code = train('sigterm', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0')
    check('2 SIGTERM: resumed run exits 0', code == 0)
    compare('ref1', 'sigterm', '2 SIGTERM')

    # 3. epoch boundary with half-accumulated gradients (accum 2, 5 micro-batches per epoch)
    code = train('accum', '--max-micro-batches-per-epoch', '5', '--grad-accum', '2', '--ckpt-minutes', '999',
                 '--crash-at-step', '4')
    check('3 accum: crashed in epoch 2', code == 1)
    code = train('accum', '--max-micro-batches-per-epoch', '5', '--grad-accum', '2', '--ckpt-minutes', '999')
    check('3 accum: resumed run exits 0', code == 0)
    check('3 accum: resumed from the epoch-1-end checkpoint', 'epoch 2, micro-batch 0' in stdout('accum'))
    compare('ref2', 'accum', '3 accum (pending gradients restored)')

    # 4 + 5. corrupted latest.pt and config mismatch
    code = train('corrupt', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0', '--crash-at-step', '8')
    latest = os.path.join(OUT, 'corrupt', 'ckpt', 'latest.pt')
    before = os.path.getsize(latest)
    code = train('corrupt', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0',
                 '--micro-batch-samples', '400000')
    check('5 mismatch: resume with a different micro-batch size is refused', code != 0 and
          'refusing to resume' in stdout('corrupt'), f'exit {code}')
    check('5 mismatch: checkpoint untouched by the refused run', os.path.getsize(latest) == before)
    with open(latest, 'r+b') as f:
        f.truncate(1000)
    code = train('corrupt', '--max-micro-batches-per-epoch', '6', '--ckpt-minutes', '0')
    check('4 corrupt: unreadable latest.pt is reported', 'could not read' in stdout('corrupt'))
    check('4 corrupt: resumed from prev.pt (step 6)', 'prev.pt: epoch 2, micro-batch 0' in stdout('corrupt')
          or 'optimizer step 6' in stdout('corrupt'))
    check('4 corrupt: resumed run exits 0', code == 0)
    compare('ref1', 'corrupt', '4 corrupt')

    print(f'{len(failures)} failures; {time.time() - t0:.0f} s')
    print('ALL RESUME CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
