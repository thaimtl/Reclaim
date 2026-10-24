"""Camera-only baselines on our wearer's recordings, run in their own environment (venv-camera).

    camera/mouth.py                 video -> 96x96 mouth crops at 25 fps + mouth opening, cached per utterance
    camera/vsr_worker.py            one lip-reading system over the cache -> N-best with scores and timings
                                    (auto_avsr, braven, liplearner embeddings); one process per system because
                                    Lipflow and BRAVEn vendor different copies of espnet
    camera/llm.py                   local LLM: Lipflow-style cleanup and N-best rescoring
    camera/run_camera_baselines.py  orchestrates everything and writes benchmark_results/CAM_*
    camera/download_weights.sh      all weights
    requirements-camera.txt         the environment
"""
