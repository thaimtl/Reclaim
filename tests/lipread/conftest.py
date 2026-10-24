"""Shared fixtures for the live engine tests.

    venv-live/bin/python -m pytest tests/lipread -q

Pure-logic tests (endpoint, run) need nothing. Tests that need weights, the sample video or a GPU skip
with the reason when those are missing.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lipread  # noqa: E402  (puts Lipflow on sys.path, sets LIPFLOW_HOME)
from lipread.types import FPS  # noqa: E402

WEIGHTS = [os.path.join(lipread.LIPFLOW_MODELS, p) for p in ('vsr/model.pth', 'vsr/model.json', 'lm/model.pth',
                                                          'lm/model.json', 'face_landmarker.task')]


def have_weights():
    return all(os.path.exists(p) for p in WEIGHTS)


def have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


needs_weights = pytest.mark.skipif(not have_weights(), reason='weights missing: HF part of camera/download_weights.sh')
needs_sample = pytest.mark.skipif(not os.path.exists(lipread.SAMPLE_VIDEO),
                                  reason='sample missing: third_party/lipflow/scripts/download-models.sh --samples')
needs_cuda = pytest.mark.skipif(not have_cuda(), reason='no CUDA GPU')


@pytest.fixture(scope='session')
def sample_video():
    if not os.path.exists(lipread.SAMPLE_VIDEO):
        pytest.skip('sample missing: third_party/lipflow/scripts/download-models.sh --samples')
    return lipread.SAMPLE_VIDEO


@pytest.fixture(scope='session')
def face_model():
    path = os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task')
    if not os.path.exists(path):
        pytest.skip('face landmarker missing: HF part of camera/download_weights.sh')
    return path


@pytest.fixture
def fps():
    return FPS
