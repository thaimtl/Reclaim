"""Shared fixtures for the live engine tests.

    venv-live/bin/python -m pytest tests/live -q

Pure-logic tests (commit, endpoint, bus) need nothing. Tests that need weights, the sample video or a GPU skip
with the reason when those are missing.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import live  # noqa: E402  (puts Lipflow on sys.path, sets LIPFLOW_HOME)
from live.types import FPS, LiveHyp, WordEnd  # noqa: E402

WEIGHTS = [os.path.join(live.LIPFLOW_MODELS, p) for p in ('vsr/model.pth', 'vsr/model.json', 'lm/model.pth',
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
needs_sample = pytest.mark.skipif(not os.path.exists(live.SAMPLE_VIDEO),
                                  reason='sample missing: third_party/lipflow/scripts/download-models.sh --samples')
needs_cuda = pytest.mark.skipif(not have_cuda(), reason='no CUDA GPU')


@pytest.fixture(scope='session')
def sample_video():
    if not os.path.exists(live.SAMPLE_VIDEO):
        pytest.skip('sample missing: third_party/lipflow/scripts/download-models.sh --samples')
    return live.SAMPLE_VIDEO


@pytest.fixture(scope='session')
def face_model():
    path = os.path.join(live.LIPFLOW_MODELS, 'face_landmarker.task')
    if not os.path.exists(path):
        pytest.skip('face landmarker missing: HF part of camera/download_weights.sh')
    return path


def make_hyp(words, n_frames, utt_id=1, t0=0.0, trailing_blanks=0):
    """LiveHyp from [(word, end_frame), ...], for synthetic commit/endpoint tests."""
    return LiveHyp(utt_id=utt_id, t0=t0, n_frames=n_frames,
                   words=tuple(WordEnd(w, e) for w, e in words), trailing_blanks=trailing_blanks)


@pytest.fixture
def hyp():
    return make_hyp


@pytest.fixture
def fps():
    return FPS
