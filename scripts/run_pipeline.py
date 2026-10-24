"""Start the whole pipeline with one command, each part in its own venv, and keep it running.

    python scripts/run_pipeline.py --rig-port COM5 --dial +14155550142      # the demo
    python scripts/run_pipeline.py --no-emg --no-call                       # camera + decoder + Claude only
    python scripts/run_pipeline.py --dry-run                                # print the commands, start nothing

Order: the logger first (everything is logged), then the camera reader, the rig publisher and the EMG reader, the
decoder, the Claude service, and the call last. Each part's output goes to the console with its name in front and to
$RECLAIM_WORKDIR/logs/run-<time>/<part>.log. A part that exits is restarted after RESTART_S, up to MAX_RESTARTS
times (then the rest keeps going without it: the decoder runs on whatever readers are left). Ctrl-C stops the parts
in reverse order (SIGTERM, or CTRL_BREAK on Windows), then kills whatever is still running after STOP_S.

Interpreters: RECLAIM_PY_<PART> (e.g. RECLAIM_PY_SVR) overrides one part's python; otherwise venv-live for the
logger, camera, rig, decoder and Claude, venv-svr for the EMG reader and venv-call for the call, all under
$RECLAIM_WORKDIR (scripts/setup_envs.sh, lipread/setup.sh, cluster/setup.sh build them). Only the standard library
is used here, so any python can run this script.

The EMG reader runs only with --rig-port, and joins the decoder only if the checkpoint passed the replay check
(docs/BUILD_PLAN.md); --no-emg leaves both out and starts the decoder with --no-emg.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.name == 'nt'
RESTART_S = 2.0
MAX_RESTARTS = 5
STOP_S = 5.0


def python_of(venv, part, work):
    override = os.environ.get(f'RECLAIM_PY_{part.upper()}')
    if override:
        return override
    sub = ('Scripts', 'python.exe') if WINDOWS else ('bin', 'python')
    return os.path.join(work, venv, *sub)


def plan(args, work):
    """[(name, argv)] in start order."""
    py = lambda venv, part: python_of(venv, part, work)
    parts = [('logger', [py('venv-live', 'logger'), '-m', 'buslog.record']),
             ('camera', [py('venv-live', 'camera'), '-m', 'lipread.run', *(['--cam', str(args.cam)]
                                                                            if args.replay is None
                                                                            else ['--replay', args.replay])])]
    if not args.no_emg and args.rig_port:
        parts.append(('rig', [py('venv-live', 'rig'), '-m', 'rig.publish', args.rig_port]))
        svr = [py('venv-svr', 'svr'), '-m', 'svr.serve', '--ckpt', args.ckpt]
        svr += ['--gains', args.gains] if args.gains else ['--calibrate', str(args.calibrate)] if args.calibrate else []
        parts.append(('svr', svr))
    parts.append(('decoder', [py('venv-live', 'decoder'), '-m', 'fusion.run',
                              *(['--no-emg'] if args.no_emg or not args.rig_port else [])]))
    if not args.no_claude:
        parts.append(('claude', [py('venv-live', 'claude'), '-m', 'claude_service.service']))
    if not args.no_call:
        parts.append(('call', [py('venv-call', 'call'), '-m', 'call.server', *(['--dial', args.dial] if args.dial
                                                                                  else [])]))
    return parts


class Part:
    def __init__(self, name, argv, log_dir, echo=print, env=None):
        self.name, self.argv, self.echo = name, argv, echo
        self.env = env or dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''),
                               PYTHONUNBUFFERED='1')
        self.log = open(os.path.join(log_dir, f'{name}.log'), 'a', encoding='utf-8', errors='replace')
        self.proc, self.restarts, self.gave_up = None, 0, False

    def start(self):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
        self.proc = subprocess.Popen(self.argv, cwd=REPO, env=self.env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1,
                                     encoding='utf-8', errors='replace', creationflags=flags)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self.echo(f'[run] {self.name}: started (pid {self.proc.pid})')

    def _pump(self, proc):
        for line in proc.stdout:
            self.log.write(line)
            self.log.flush()
            self.echo(f'[{self.name}] {line.rstrip()}')

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.running():
            try:
                if WINDOWS:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except OSError:
                pass

    def kill(self):
        if self.running():
            self.proc.kill()

    def close(self):
        self.log.close()


def supervise(parts, stop, restart_s=RESTART_S, max_restarts=MAX_RESTARTS, poll_s=0.2, echo=print):
    """Start the parts in order, restart one that exits, until stop() is true; then stop them in reverse order."""
    for p in parts:
        p.start()
    died = {}
    try:
        while not stop():
            now = time.monotonic()
            for p in parts:
                if p.gave_up or p.running():
                    continue
                if p.name not in died:
                    died[p.name] = now
                    echo(f'[run] {p.name}: exited with {p.proc.returncode}')
                elif now - died[p.name] >= restart_s:
                    del died[p.name]
                    if p.restarts >= max_restarts:
                        p.gave_up = True
                        echo(f'[run] {p.name}: gave up after {max_restarts} restarts, the rest goes on')
                    else:
                        p.restarts += 1
                        p.start()
            time.sleep(poll_s)
    finally:
        shutdown(parts, echo=echo)


def shutdown(parts, stop_s=STOP_S, echo=print):
    for p in reversed(parts):
        p.stop()
    end = time.monotonic() + stop_s
    while any(p.running() for p in parts) and time.monotonic() < end:
        time.sleep(0.1)
    for p in parts:
        if p.running():
            echo(f'[run] {p.name}: still running after {stop_s:.0f} s, killed')
            p.kill()
        p.close()


def main(argv=None):
    work = os.environ.get('RECLAIM_WORKDIR', REPO)
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cam', type=int, default=0, help='webcam index')
    ap.add_argument('--replay', help='a recorded video instead of the webcam (lipread.run --replay)')
    ap.add_argument('--rig-port', help="the Uno's serial port (COM5, /dev/ttyACM0); without it, no EMG")
    ap.add_argument('--ckpt', default=os.path.join(work, 'checkpoints', 'W_svr1', 'model.pt'))
    ap.add_argument('--calibrate', type=float, default=0.0, metavar='S', help='EMG calibration seconds at start')
    ap.add_argument('--gains', help='a saved EMG calibration (svr.serve --calibrate wrote it)')
    ap.add_argument('--dial', help='call this number once the call server is up (E.164)')
    ap.add_argument('--no-emg', action='store_true')
    ap.add_argument('--no-claude', action='store_true')
    ap.add_argument('--no-call', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    parts = plan(args, work)
    if args.dry_run:
        for name, cmd in parts:
            print(f'{name:8s} {" ".join(cmd)}')
        return parts
    missing = [cmd[0] for _, cmd in parts if not os.path.exists(cmd[0])]
    if missing:
        ap.error('missing interpreters (build the venvs or set RECLAIM_PY_<PART>): ' + ', '.join(sorted(set(missing))))
    log_dir = os.path.join(work, 'logs', time.strftime('run-%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    print(f'[run] logs in {log_dir}; Ctrl-C stops everything')
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        supervise([Part(n, c, log_dir) for n, c in parts], stopping.is_set)
    except KeyboardInterrupt:
        pass
    return parts


if __name__ == '__main__':
    main()
