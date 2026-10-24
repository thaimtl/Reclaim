"""B1: Gaddy's released weights in our reimplementation (svr), on CPU and on GPU.

Same outputs as b0_dump.py (inputs, per-frame log-probs, timings), but everything comes
from our code: svr.preprocess for the input, svr.model for the network. Beam search is run
later by score.py with our decoder.

    venv/bin/python -m bench.b1_run
"""
from bench.common import GADDY_CHECKPOINT
from bench.run_model import run


def main():
    run('B1_ours', GADDY_CHECKPOINT,
        "Gaddy's released recognition checkpoint (Zenodo 7183877) loaded into our "
        'reimplementation (svr), our preprocessing, current environment')


if __name__ == '__main__':
    main()
