"""scripts/run_pipeline.py: which parts start with which flags, and the supervisor (restart, give up, logs, stop)
with tiny fake parts instead of the real ones."""
import importlib.util
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('run_pipeline', os.path.join(ROOT, 'scripts', 'run_pipeline.py'))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def names(argv):
    return [(n, cmd[1:]) for n, cmd in R.main(['--dry-run', *argv])]


def test_demo_plan_starts_everything_in_order(capsys):
    got = dict(names(['--rig-port', 'COM5', '--dial', '+14155550142', '--calibrate', '60']))
    assert list(got) == ['logger', 'camera', 'rig', 'svr', 'decoder', 'claude', 'call']
    assert got['rig'] == ['-m', 'rig.publish', 'COM5'] and got['decoder'] == ['-m', 'fusion.run']
    assert got['svr'][-2:] == ['--calibrate', '60.0'] and got['call'] == ['-m', 'call.server', '--dial', '+14155550142']
    assert 'decoder' in capsys.readouterr().out


def test_without_a_rig_the_decoder_runs_camera_only():
    got = dict(names(['--no-call', '--no-claude']))
    assert list(got) == ['logger', 'camera', 'decoder'] and got['decoder'] == ['-m', 'fusion.run', '--no-emg']
    got = dict(names(['--rig-port', 'COM5', '--no-emg', '--replay', 'clip.mp4']))
    assert 'svr' not in got and 'rig' not in got and got['camera'][-2:] == ['--replay', 'clip.mp4']


def test_interpreter_override(monkeypatch):
    monkeypatch.setenv('RECLAIM_PY_SVR', '/opt/svr/python')
    assert R.python_of('venv-svr', 'svr', '/w') == '/opt/svr/python'
    monkeypatch.delenv('RECLAIM_PY_SVR')
    assert R.python_of('venv-svr', 'svr', '/w').startswith(os.path.join('/w', 'venv-svr'))


def test_supervisor_restarts_gives_up_and_stops(tmp_path):
    lines = []
    flaky = R.Part('flaky', [sys.executable, '-c', 'print("boom")'], str(tmp_path), echo=lines.append)
    steady = R.Part('steady', [sys.executable, '-c', 'import time\nprint("up", flush=True)\ntime.sleep(60)'],
                    str(tmp_path), echo=lines.append)
    stop = threading.Event()
    th = threading.Thread(target=R.supervise, args=([flaky, steady], stop.is_set),
                          kwargs=dict(restart_s=0.05, max_restarts=2, poll_s=0.02, echo=lines.append))
    th.start()
    deadline = time.monotonic() + 10
    while not flaky.gave_up and time.monotonic() < deadline:
        time.sleep(0.05)
    assert flaky.gave_up and flaky.restarts == 2 and steady.running()
    stop.set()
    th.join(10)
    assert not th.is_alive() and not steady.running()
    assert (tmp_path / 'flaky.log').read_text().count('boom') == 3      # first run + 2 restarts
    assert '[steady] up' in lines and any('gave up' in x for x in lines)
