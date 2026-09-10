# Release model assets

The two Transmelody checkpoints contain only model tensors, architecture / feature
configuration, minimal inference defaults, and non-personal release provenance.
Private paths, song labels, optimizer state, and detailed training logs were removed.
Every state tensor was checked for exact equality with the current working model.
See `manifest.json` for file SHA-256 hashes and source checkpoint identities.

- `transmelody_base.pt`: complete-note segmentation and baseline pitch model.
- `transmelody_pitch.pt`: auxiliary pitch Transformer; frozen in the normal workflow.

The model files are publicly provided for evaluation. The MIT grant in the repository
applies to original source code; a separate affirmative model-weight usage license
has not yet been selected. Do not assume permission for model redistribution or
commercial model use solely from the source-code license. Third-party dependencies
retain their own terms. No rights to the underlying musical works are conveyed.

No source audio, annotated MIDI, lyrics, dataset registry or feature cache is included.
These are small-data experimental weights, not a general-purpose transcription benchmark.

Run `python scripts/setup_models.py` from the repository root to configure inference.
