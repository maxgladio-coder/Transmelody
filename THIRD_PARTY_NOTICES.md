# Third-party components

This repository's original-source MIT license does not replace upstream licenses.

| Component | Role | Distribution / source |
| --- | --- | --- |
| Beat This! | Beat and downbeat tracking | `beat-this` package; checkpoint downloaded on first use. [Upstream](https://github.com/CPJKU/beat_this) states code and published weights are MIT. |
| Japanese wav2vec2 | Frozen pronunciation-context features | Downloaded from [Reazon Research](https://huggingface.co/reazon-research/japanese-wav2vec2-base-rs35kh); model card declares Apache-2.0. |
| RVC RMVPE runtime | Continuous pitch evidence | Vendored `rmvpe.py` and `tools/cuda_graph.py` from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), with original MIT notice in `third_party/rmvpe/LICENSE`. Runtime files are unchanged. |
| RMVPE pretrained weight | Pitch-evidence network | Not redistributed in Git. Setup downloads [upstream rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) and verifies its known SHA-256. Review upstream model terms before use. |
| PyTorch, torchaudio, NumPy, SciPy, librosa, soundfile, matplotlib, mido, Transformers, safetensors, openpyxl | Numerical / model / file runtime | Installed through requirements; see each distribution for license notices. |
| FFmpeg | Optional audio-conversion fallback | Not bundled; users install an appropriate build and comply with its terms. |

RMVPE research: [RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://arxiv.org/abs/2306.15412).
This release uses the RVC implementation; it does not claim authorship of RMVPE.

Optional SOFA / Japanese lyric-alignment experiments require separate third-party
source, weights, and dependencies. They are not downloaded by setup_models.py and
are not required for default Transmelody prediction.

The default Japanese encoder is pinned to revision
`46afc596052b612293c8db256b3a69447a2f57dc`, matching the tested local cache.
The unchanged vendored RMVPE demonstration block includes its upstream author's
example Windows paths; these are not Transmelody user data and are not executed
by the production import path.
