"""Optional real GPU integration smoke; synthetic sound tests execution, NOT accuracy.

Run with RUN_SOFA_SMOKE=1 after installing the optional model/dependencies.
"""
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from transmelody.lyrics.lyric_alignment import align


@pytest.mark.skipif(os.environ.get("RUN_SOFA_SMOKE") != "1", reason="Optional downloaded SOFA model")
def test_real_sofa_safe_checkpoint_and_inference(tmp_path):
    sr = 44100
    t = np.arange(sr * 3) / sr
    wave = (.1 * np.sin(2 * np.pi * 220 * t) + .03 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "synthetic.wav"
    sf.write(path, wave, sr)
    data = align(path, {"song_id": "99999", "phrases": [{"text": "かな", "units": [["k", "a"], ["n", "a"]],
        "start": .5, "end": 2.5}]}, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert data["status"] == "candidate" and data["reviewed_regions"] == []
    assert len(data["units"]) == 2 and len(data["phones"]) == 4
    assert all(.5 <= u["start"] < u["end"] <= 2.5 for u in data["units"])
