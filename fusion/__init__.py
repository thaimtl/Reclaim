"""The fusion decoder: camera ticks (later EMG) + Claude's predictions -> locked words, captions and voice.

    words.py    cam.tick -> words with frame spans, camera confidence and alternatives (CTC greedy)
    lock.py     one lock score per word from a feature dict, and the moving threshold
    decoder.py  the state machine: utterances, pinned locked prefix, repair window, sentence end, "please repeat"
    run.py      the bus loop: python -m fusion.run
"""
