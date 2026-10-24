"""Silent Voice Rig: our reimplementation of the Gaddy EMG-to-text model.

Modules:
    text        character vocabulary and text normalization
    preprocess  raw EMG -> model input (filters, resampling, scaling)
    model       ResNet + relative-position Transformer encoder with a CTC head
    decoders    greedy CTC, beam search with a KenLM word LM, phrase scoring
"""
