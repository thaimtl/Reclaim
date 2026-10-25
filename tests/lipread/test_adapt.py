"""The face adapter: lipread/adapt.py's data handling, and LiveModel loading an adapter over the base weights."""
import json

import numpy as np
import pytest

from lipread.adapt import main, norm, sentence_clips, split

from .conftest import needs_cuda, needs_weights


def test_norm_matches_how_wer_compares():
    assert norm("I'm a well-known guy, okay?") == "I'M A WELL KNOWN GUY OKAY"


def test_split_holds_out_every_nth_clip():
    rows = list(range(10))
    train, test = split(rows, 5)
    assert test == [4, 9] and train == [0, 1, 2, 3, 5, 6, 7, 8]
    assert split(rows, 0) == (rows, [])


def test_sentence_clips_skip_rest_clips_and_sort(tmp_path):
    d = tmp_path / 'session_01'
    d.mkdir()
    rows = [dict(id='s1_002', session='session_01', prompt='Second.'), dict(id='s1_rest_01', session='session_01',
            prompt=''), dict(id='s1_001', session='session_01', prompt='First.')]
    (d / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert [r['id'] for r in sentence_clips(['session_01'], root=str(tmp_path))] == ['s1_001', 's1_002']


def test_sentence_clips_read_from_the_folder_given(tmp_path):
    d = tmp_path / 'session_02_EMG-dropped-frames'
    d.mkdir()
    (d / 'manifest.jsonl').write_text(json.dumps(dict(id='s2_001', session='session_02', prompt='Hi.')) + '\n')
    assert [r['dir'] for r in sentence_clips([d.name], root=str(tmp_path))] == [d.name]


def test_never_trains_on_session_3():
    with pytest.raises(SystemExit):
        main(['--sessions', 'session_01', 'session_03'])


def test_a_session_is_not_both_trained_on_and_held_out():
    with pytest.raises(SystemExit):
        main(['--sessions', 'session_01', '--test-sessions', 'session_01'])


@pytest.fixture(scope='module')
def adapter_file(tmp_path_factory):
    """A face adapter with the right tensors and changed values (as lipread/adapt.py saves one)."""
    import torch
    from lipflow.vsr import LipReader
    from lipread.model import FACE_ADAPTER_PREFIXES
    reader = LipReader(device='cpu', use_lm=False, personal=False)
    g = torch.Generator().manual_seed(0)
    state = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in reader.model.state_dict().items()
             if k.startswith(FACE_ADAPTER_PREFIXES) and v.is_floating_point()}
    path = tmp_path_factory.mktemp('adapter') / 'face.pth'
    torch.save(state, path)
    return path, state


@needs_weights
@needs_cuda
def test_live_model_loads_the_adapter_and_reads_differently(adapter_file):
    import torch
    from lipread.model import LiveModel
    path, state = adapter_file
    base = LiveModel(warmup=())
    adapted = LiveModel(warmup=(), face_adapter=str(path))
    sd = adapted.reader.model.state_dict()
    assert all(torch.equal(sd[k].cpu(), v) for k, v in state.items())
    other = [k for k in sd if k.startswith('encoder.encoders.5.')]
    assert other and all(torch.equal(sd[k], base.reader.model.state_dict()[k]) for k in other)
    rois = np.random.default_rng(0).integers(0, 256, (30, 96, 96), dtype=np.uint8)
    a, b = base.read(rois), adapted.read(rois)
    assert a.topk_logp.shape == b.topk_logp.shape and not np.allclose(a.topk_logp, b.topk_logp)


@needs_weights
def test_a_file_with_other_tensors_is_refused(tmp_path, adapter_file):
    import torch
    from lipflow.vsr import LipReader
    from lipread.model import load_face_adapter
    _, state = adapter_file
    bad = dict(state)
    bad['decoder.embed.0.weight'] = torch.zeros(1)
    torch.save(bad, tmp_path / 'bad.pth')
    reader = LipReader(device='cpu', use_lm=False, personal=False)
    with pytest.raises(ValueError, match='not a face adapter'):
        load_face_adapter(reader.model, str(tmp_path / 'bad.pth'))
