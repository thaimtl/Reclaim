"""B0: Gaddy's released model, run with his own code in his exact environment, on CPU.

For every recording in the frozen splits, at 8 and 4 channels, saves:
    - the model input (his load_utterance + the scaling in his EMGDataset.__getitem__)
    - per-frame log-probabilities (his Model, his checkpoint)
    - his ctcdecode beam-search transcript (open vocab only; alpha 1.5, beta 1.85, beam 100)
    - CPU timings
4 channels uses his own --remove_channels mechanism (zeroing channels 3, 4, 6, 7).

Run from baseline_gaddy/silent_speech with the B0 env:
    ../.venv/bin/python ../../bench/b0_dump.py
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # PyTorch 2.0 has no kernels for the RTX 5090: CPU only

sys.path.insert(0, os.getcwd())  # Gaddy's modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from absl import flags  # noqa: E402

import read_emg  # noqa: E402
from architecture import Model  # noqa: E402
from data_utils import TextTransform  # noqa: E402

from bench.common import (CHANNEL_CONFIGS, CLIP_SECONDS, EVAL_SPLITS, FRAME_RATE, GADDY_CHECKPOINT,  # noqa: E402
                          RESULTS_DIR, Timer, environment_info, latency_stats, load_split, md5, save_raw)

OUT_DIR = os.path.join(RESULTS_DIR, 'B0_gaddy_exactenv_cpu', 'raw')
WARMUP, REPEATS = 5, 30


def his_model_input(session_dir, index):
    _, emg_features, _, _, _, raw_emg = read_emg.load_utterance(os.path.join('emg_data', session_dir), index)
    raw_emg = raw_emg / 20  # EMGDataset.__getitem__
    raw_emg = 50 * np.tanh(raw_emg / 50.)
    return raw_emg, emg_features


def main():
    flags.FLAGS(['b0_dump'])
    import ctcdecode
    text_transform = TextTransform()
    n_chars = len(text_transform.chars)
    model = Model(0, n_chars + 1)  # num_features is unused by his Model
    model.load_state_dict(torch.load(GADDY_CHECKPOINT, map_location='cpu'))
    model.eval()
    decoder = ctcdecode.CTCBeamDecoder(text_transform.chars + '_', blank_id=n_chars, log_probs_input=True,
                                       model_path='lm.binary', alpha=1.5, beta=1.85)

    def forward(x_raw):
        x_feat = torch.zeros(1, x_raw.shape[1] // 8, 112)
        sess = torch.zeros(1, x_raw.shape[1] // 8, dtype=torch.long)
        return F.log_softmax(model(x_feat, x_raw, sess), -1)

    meta_common = {
        'model': 'B0_gaddy_exactenv_cpu',
        'description': "Gaddy's released recognition checkpoint (Zenodo 7183877), his code and exact environment",
        'checkpoint': GADDY_CHECKPOINT, 'checkpoint_md5': md5(GADDY_CHECKPOINT),
        'device': 'cpu', 'environment': environment_info(torch),
        'ctcdecode': {'version': getattr(ctcdecode, '__version__', '1.0.3 (git c90ad94)'), 'alpha': 1.5, 'beta': 1.85,
                      'beam_width': 100, 'cutoff_top_n': 40, 'lm': 'DeepSpeech v0.6.1 lm.binary'},
        'parameters': sum(p.numel() for p in model.parameters()),
    }

    with torch.no_grad():
        for channels, cfg in CHANNEL_CONFIGS.items():
            drop = [c for c in range(8) if c not in cfg['keep_channels']]
            flags.FLAGS.remove_channels = [str(c) for c in drop]

            # latency on a fixed 3 s clip (first open-vocab test recording, cropped)
            first = load_split('open_vocab_test')['items'][0]
            clip_frames = int(round(CLIP_SECONDS * FRAME_RATE))
            clip = torch.from_numpy(his_model_input(first['session_dir'], first['index'])[0][:clip_frames * 8])[None]
            for _ in range(WARMUP):
                forward(clip)
            clip_ms = []
            for _ in range(REPEATS):
                with Timer() as t:
                    forward(clip)
                clip_ms.append(t.ms)

            for split in EVAL_SPLITS:
                items_out, logits, inputs = [], {}, {}
                for it in load_split(split)['items']:
                    with Timer() as t_pre:
                        raw_emg, _ = his_model_input(it['session_dir'], it['index'])
                    x_raw = torch.from_numpy(raw_emg)[None]
                    with Timer() as t_fwd:
                        lp = forward(x_raw)
                    row = {'id': it['id'], 'n_frames': int(lp.shape[1]),
                           'duration_s': raw_emg.shape[0] / 689.06,
                           'preprocess_ms': t_pre.ms, 'forward_ms': t_fwd.ms}
                    if split == 'open_vocab_test':
                        with Timer() as t_beam:
                            beam_results, _, _, out_lens = decoder.decode(lp)
                        ids = beam_results[0, 0, :out_lens[0, 0]].tolist()
                        row['ctcdecode_text'] = text_transform.int_to_text(ids)
                        row['ctcdecode_ms'] = t_beam.ms
                    items_out.append(row)
                    logits[it['id']] = lp[0].numpy().astype(np.float32)
                    inputs[it['id']] = raw_emg.astype(np.float32)
                meta = dict(meta_common, split=split, channels=channels, channel_config=cfg,
                            remove_channels_flag=flags.FLAGS.remove_channels,
                            latency_3s_clip_forward_ms=latency_stats(clip_ms),
                            notes='preprocess_ms runs his full load_utterance, which also computes hand-crafted '
                                  'EMG features and the audio mel spectrogram that the model does not use.')
                save_raw(OUT_DIR, split, channels, meta, items_out, logits, inputs)
                print(split, channels, len(items_out), 'done', flush=True)


if __name__ == '__main__':
    main()
