"""One-screen status of a supervised training run.

    venv/bin/python -m svr.status B1T
"""
import glob
import json
import os
import sys
import time

import psutil
import pynvml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo


def main():
    name = sys.argv[1]
    run = os.path.join(WORK, 'checkpoints', name)
    with open(os.path.join(run, 'config.json')) as f:
        cfg = json.load(f)
    epochs = {}
    if os.path.exists(os.path.join(run, 'log.jsonl')):
        for line in open(os.path.join(run, 'log.jsonl')):
            r = json.loads(line)
            epochs[r['epoch']] = r  # a resume may re-log an epoch; keep the latest
    rows = [epochs[k] for k in sorted(epochs)]
    out = [f'== {name}  ({time.strftime("%Y-%m-%d %H:%M:%S")})']
    if rows:
        last = rows[-1]
        recent = rows[-5:]
        epoch_s = sum(r.get('epoch_wall_seconds', 0) for r in recent) / len(recent)
        remaining = cfg['epochs'] - last['epoch']
        beams = [r for r in rows if 'dev_wer_beam_lm' in r]
        losses = '  '.join(f"{k[6:]} {v:.3f}" for k, v in last.items() if k.startswith('train_') and k != 'train_seconds')
        out.append(f"epoch {last['epoch']}/{cfg['epochs']}  train {losses}  "
                   f"dev WER greedy {100 * last['dev_wer_greedy']:.1f}%"
                   + (f"  beam+LM {100 * beams[-1]['dev_wer_beam_lm']:.1f}% (epoch {beams[-1]['epoch']})" if beams else '')
                   + (f"  dev homophene-10 {100 * last['dev_homophene10_top1']:.1f}%"
                      if 'dev_homophene10_top1' in last else ''))
        out.append(f"epoch time {epoch_s:.0f} s  ETA {remaining * epoch_s / 3600:.2f} h  lr {last['lr']:.2e}  "
                   f"optimizer step {last['opt_step']}")
        tflops = [r['achieved_tflops'] for r in recent if 'achieved_tflops' in r]
        out.append(f"GPU-hours {last['gpu_hours_total']:.2f}  "
                   + (f"achieved {last['achieved_tflops']:.1f} TFLOPS (avg last 5: {sum(tflops) / len(tflops):.1f})  "
                      if tflops and 'achieved_tflops' in last else '')
                   + f"peak VRAM (PyTorch) {max(r['peak_vram_reserved_gb'] for r in rows):.1f} GB reserved")
    progress = os.path.join(run, 'progress.jsonl')
    if os.path.exists(progress):
        p = json.loads(open(progress).read().strip().split('\n')[-1])
        out.append(f"in-epoch: epoch {p['epoch']} micro-batch {p['micro_batch']}/{p['micro_batches_in_epoch']}  "
                   f"recent loss {p['recent_loss']:.3f}  ({time.time() - p['time']:.0f} s ago)")
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    out.append(f"now: GPU {pynvml.nvmlDeviceGetUtilizationRates(h).gpu}%  VRAM {mem.used / 2 ** 30:.1f}/"
               f"{mem.total / 2 ** 30:.1f} GB  {pynvml.nvmlDeviceGetPowerUsage(h) / 1000:.0f} W  "
               f"{pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)} C  |  "
               f"RAM {psutil.virtual_memory().percent:.0f}%  CPU {psutil.cpu_percent(interval=1):.0f}%")
    trainers = [p for p in psutil.process_iter(['cmdline', 'ppid'])
                if 'svr.train' in ' '.join(p.info['cmdline'] or []) and f'--name {name}' in ' '.join(p.info['cmdline'] or [])]
    out.append(f"processes: {len(trainers)} (trainer + Erbium monitor = 2 expected)  "
               f"orphans: {sum(p.info['ppid'] == 1 for p in trainers)}")
    reports = sorted(glob.glob(os.path.join(run, 'erbium', '*.txt')))
    if reports:
        section, keep = '', []
        for line in open(reports[-1]).read().splitlines():
            s = line.strip()
            if s in ('CPUs', 'GPUs'):
                section = 'CPU' if s == 'CPUs' else 'GPU'
            elif s.startswith('Duration'):
                keep.append(s)
            elif section and 'tilization (%)' in s:
                kind = 'RAM' if (section == 'CPU' and s.startswith('Memory')) else \
                       'VRAM' if s.startswith('Memory') else section
                keep.append(f"{kind} {s.split(':', 1)[1].strip()}")
        out.append('Erbium (whole run, %): ' + ' | '.join(keep))
    sup = os.path.join(run, 'supervisor.log')
    if os.path.exists(sup):
        lines = [l.strip() for l in open(sup) if l.strip()]
        crashes = sum('crashed' in l for l in lines)
        out.append(f"supervisor: {lines[-1] if lines else ''}  (crashes so far: {crashes})")
    if os.path.exists(os.path.join(run, 'DONE')):
        out.append('DONE: ' + open(os.path.join(run, 'DONE')).read().strip())
    print('\n'.join(out))


if __name__ == '__main__':
    main()
