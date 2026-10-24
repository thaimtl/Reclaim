"""Live lip-reading engine for Reclaim free speech (plan: docs/EVAN_REALTIME_LIP_READING.md, docs/REALTIME_PIPELINE.md).

Webcam -> mouth crops -> Auto-AVSR encoder re-reads the current segment every 100 ms -> cam.tick / cam.final on the
team bus (ZeroMQ + msgpack, bus/__init__.py, bus/schema.md).
Reuses Lipflow (third_party/lipflow, MIT) by import; importing this package puts it on sys.path.
Data (third_party/, data/, recordings/) is found through RECLAIM_WORKDIR, as for the rest of the repo (README,
"Code and data"); without it, next to the code.

Runs in venv-live (Python 3.11, requirements-camera.txt pins + torch 2.11 cu130):
    venv-live/bin/python -m lipread.run --replay third_party/lipflow/samples/2016-03-12.mov
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
LIPFLOW = os.path.join(WORK, 'third_party', 'lipflow')
LIPFLOW_MODELS = os.path.join(LIPFLOW, 'models')
SAMPLE_VIDEO = os.path.join(LIPFLOW, 'samples', '2016-03-12.mov')
SAMPLE_START, SAMPLE_END = 20.4, 28.1
SAMPLE_TEXT = 'BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE'

if LIPFLOW not in sys.path:
    sys.path.insert(0, LIPFLOW)
os.environ.setdefault('LIPFLOW_HOME', os.path.join(WORK, 'data', 'lipflow_home'))
