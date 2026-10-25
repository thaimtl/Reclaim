"""Runs tests/smoke_claude_service.py (its offline checks) under pytest, so the Claude service is tested with the rest."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_claude_service_smoke():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'smoke_claude_service.py')], cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, [line for line in r.stdout.splitlines() if line.startswith('FAIL')] or r.stderr[-2000:]
