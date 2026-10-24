"""Run ONE camera system over the cached mouth crops and write its raw outputs.

    venv-camera/bin/python -m camera.vsr_worker --system auto_avsr  --ids ids.txt --out raw_auto_avsr.json
    venv-camera/bin/python -m camera.vsr_worker --system braven     --ids ids.txt --out raw_braven.json
    venv-camera/bin/python -m camera.vsr_worker --system liplearner --ids ids.txt --out raw_liplearner.npz

auto_avsr   Auto-AVSR LRS3 visual-only model (LRS3 WER 19.1%) + subword RNN LM, via Lipflow's LipReader, beam 20.
braven      BRAVEn Large, self-trained, LRS3+Vox2+AVS (LRS3 WER 20.9%, 20.1% with its LM), beam 40, LM weight 0.3.
liplearner  LipLearner contrastive encoder (CHI 2023): one 500-d embedding per clip, for few-shot classification.

Each system runs in its own process: Lipflow and BRAVEn vendor different copies of espnet.
Timing: encode_ms (visual encoder) and decode_ms (beam search) per utterance, after a warm-up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
THIRD = os.path.join(WORK, 'third_party')
WEIGHTS = os.path.join(WORK, 'data', 'camera_weights')
CACHE = os.path.join(WORK, 'data', 'camera_cache')
MEAN, STD = 0.421, 0.165


def load_rois(uid):
    d = np.load(os.path.join(CACHE, f'{uid}.npz'))
    return d['rois']


def center88(rois):
    o = (rois.shape[-1] - 88) // 2
    return rois[:, o:o + 88, o:o + 88]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ Auto-AVSR (through Lipflow)
class AutoAVSR:
    def __init__(self, beam_size=20, nbest=20):
        os.environ.setdefault('LIPFLOW_HOME', os.path.join(WORK, 'data', 'lipflow_home'))  # never touch ~
        sys.path.insert(0, os.path.join(THIRD, 'lipflow'))
        from lipflow.vsr import LipReader
        from espnet.asr.asr_utils import add_results_to_json
        self.add_results_to_json = add_results_to_json
        self.reader = LipReader(beam_size=beam_size, personal=False)
        self.nbest = nbest

    @torch.inference_mode()  # encode() returns an inference tensor; the beam search must run in the same mode
    def __call__(self, rois):
        sync()
        t0 = time.perf_counter()
        enc = self.reader.encode(rois)
        sync()
        t1 = time.perf_counter()
        for module in self.reader.beam.full_scorers['decoder'].modules():
            if hasattr(module, '_mem_kv'):
                del module._mem_kv
        hyps = self.reader.beam(enc)
        t2 = time.perf_counter()
        out, seen = [], set()
        for h in hyps[:self.nbest]:
            text = self.reader._clean(self.add_results_to_json([h.asdict()], self.reader.token_list)).lower()
            if text not in seen:
                seen.add(text)
                out.append({'text': text, 'score': float(h.score)})
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000


# ------------------------------------------------------------------ BRAVEn (through the raven repo)
class BRAVEn:
    CKPT = 'braven_vsr_large_lrs3vox2avs_self.pth'
    LM = 'braven_lm.pth'

    def __init__(self, beam_size=40, lm_weight=0.3, ctc_weight=0.1, nbest=20):
        raven = os.path.join(THIRD, 'raven')
        sys.path.insert(0, raven)
        from omegaconf import OmegaConf
        from espnet.asr.asr_utils import add_results_to_json, torch_load
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
        from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
        from espnet.nets.scorers.length_bonus import LengthBonus
        from utils import UNIGRAM1000_LIST
        self.add_results_to_json = add_results_to_json
        self.token_list = UNIGRAM1000_LIST
        odim = len(self.token_list)
        cfg = OmegaConf.create({'model': {
            'visual_backbone': OmegaConf.load(os.path.join(raven, 'conf/model/visual_backbone/resnet_transformer_large.yaml')),
            'language_model': OmegaConf.load(os.path.join(raven, 'conf/model/language_model/default.yaml'))}})
        backbone = OmegaConf.to_container(cfg.model.visual_backbone, resolve=True)
        self.model = E2E(odim, argparse.Namespace(**backbone))
        self.model.load_state_dict(torch.load(os.path.join(WEIGHTS, self.CKPT), map_location='cpu'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()
        scorers = self.model.scorers()
        lm = TransformerLM(odim, argparse.Namespace(**OmegaConf.to_container(cfg.model.language_model)))
        torch_load(os.path.join(WEIGHTS, self.LM), lm)
        lm.eval()
        scorers['lm'] = lm
        scorers['length_bonus'] = LengthBonus(odim)
        self.beam = BatchBeamSearch(beam_size=beam_size, vocab_size=odim, scorers=scorers,
                                    weights=dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=lm_weight,
                                                 length_bonus=0.0),
                                    sos=odim - 1, eos=odim - 1, token_list=self.token_list,
                                    pre_beam_score_key='decoder').to(self.device).eval()
        self.nbest = nbest

    @torch.inference_mode()
    def __call__(self, rois):
        x = (torch.from_numpy(center88(rois)).float() / 255.0 - MEAN) / STD
        sync()
        t0 = time.perf_counter()
        feat, _ = self.model.encoder(x.unsqueeze(0).to(self.device), None)  # frontend adds the channel dim
        sync()
        t1 = time.perf_counter()
        hyps = self.beam(x=feat.squeeze(0), maxlenratio=0.0, minlenratio=0.0)
        t2 = time.perf_counter()
        out, seen = [], set()
        for h in hyps[:self.nbest]:
            text = self.add_results_to_json([h.asdict()], self.token_list)
            text = ' '.join(text.replace('<eos>', '').replace('▁', ' ').split()).lower()
            if text not in seen:
                seen.add(text)
                out.append({'text': text, 'score': float(h.score)})
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000


# ------------------------------------------------------------------ LipLearner encoder
class LipLearner:
    CKPT = 'LipLearner_pretrained_model.pt'

    def __init__(self):
        sys.path.insert(0, os.path.join(THIRD, 'LipLearner', 'pretraining'))
        from model.model import VideoModel
        self.model = VideoModel(argparse.Namespace(n_dimention=500))
        state = torch.load(os.path.join(WEIGHTS, self.CKPT), map_location='cpu')
        state = state.get('video_model', state) if isinstance(state, dict) else state
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, rois):
        x = torch.from_numpy(center88(rois)).float()[None, :, None] / 255.0  # (1, T, 1, 88, 88)
        border = torch.ones(1, x.shape[1])  # the whole clip is the utterance
        sync()
        t0 = time.perf_counter()
        emb = self.model(x.to(self.device), border.to(self.device))[0]
        sync()
        return torch.nn.functional.normalize(emb, dim=0).cpu().numpy(), (time.perf_counter() - t0) * 1000


SYSTEMS = {'auto_avsr': AutoAVSR, 'braven': BRAVEn, 'liplearner': LipLearner}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    p.add_argument('--ids', required=True, help='text file with one utterance id per line')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    system = SYSTEMS[a.system]()
    # warm-up, not timed, on the first utterance with a face (a face-less clip has 0 ROIs and crashes conv3d)
    warm = next((r for r in map(load_rois, ids) if len(r) >= 4), None)
    if warm is not None:
        system(warm)
    if a.system == 'liplearner':
        embs, times = {}, {}
        for uid in ids:
            embs[uid], times[uid] = system(load_rois(uid))
        np.savez(a.out, ids=np.array(ids), emb=np.stack([embs[u] for u in ids]),
                 encode_ms=np.array([times[u] for u in ids]))
        return
    rows = {}
    for uid in ids:
        rois = load_rois(uid)
        if len(rois) < 4:
            rows[uid] = {'nbest': [{'text': '', 'score': 0.0}], 'encode_ms': 0.0, 'decode_ms': 0.0, 'frames': 0}
            continue
        nbest, enc_ms, dec_ms = system(rois)
        rows[uid] = {'nbest': nbest or [{'text': '', 'score': 0.0}], 'encode_ms': enc_ms, 'decode_ms': dec_ms,
                     'frames': int(len(rois))}
    meta = {'system': a.system, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'torch': torch.__version__}
    with open(a.out, 'w') as f:
        json.dump({'meta': meta, 'utterances': rows}, f)
    print('wrote', a.out, len(rows), 'utterances')


if __name__ == '__main__':
    main()
