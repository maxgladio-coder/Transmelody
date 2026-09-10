"""Withdrawn-window UI workflow smoke; enabled explicitly on a desktop host."""
import copy
import json
import os
import tkinter as tk

import numpy as np
import pytest
import soundfile as sf

from pronunciation_labels import SCHEMA, digest
from pronunciation_review_ui import ReviewWindow


@pytest.mark.skipif(os.environ.get("RUN_REVIEW_UI_SMOKE") != "1", reason="Optional desktop Tk smoke")
def test_review_edit_invalidates_confirmation_and_export_requires_review(tmp_path, monkeypatch):
    audio = tmp_path / "test.wav"
    sf.write(audio, np.sin(np.arange(16000) * .1).astype(np.float32) * .1, 16000)
    data = {"schema_version": SCHEMA, "time_reference": "original_wav_seconds",
        "status": "candidate", "song_id": "1", "audio_sha256": digest(audio), "audio_path": str(audio),
        "audio_duration": 1., "reviewed_regions": [], "phones": [],
        "phrases": [{"text": "test", "start": 0., "end": 1.}],
        "units": [{"start": .1, "end": .4, "label": "k a", "phrase": 0},
                  {"start": .4, "end": .8, "label": "n a", "phrase": 0}]}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("pronunciation_review_ui.messagebox.showinfo", lambda *a, **k: None)
    root = tk.Tk()
    root.withdraw()
    try:
        window = ReviewWindow(root, candidate, tmp_path)
        window.confirm_phrase()
        assert window.checked == {0}
        window.table.selection_set("0")
        window.select()
        window.merge_unit()
        assert window.checked == set() and len(window.data["units"]) == 1
        window.table.selection_set("0")
        window.select()
        window.cursor = .5
        window.split_unit()
        assert len(window.data["units"]) == 2
        window.table.selection_set("0")
        window.select()
        previous = copy.deepcopy(window.data["units"])
        window.start.set("nan")
        with pytest.raises(ValueError):
            window.update_unit()
        assert window.data["units"] == previous
        window.reviewer.set("tester")
        with pytest.raises(ValueError):
            window.accept()
        window.confirm_phrase()
        window.accept()
        reviewed = json.loads((tmp_path / "pronunciation_labels/1.json").read_text())
        assert reviewed["status"] == "reviewed" and reviewed["reviewed_regions"] == [[0., 1.]]
        assert json.loads(candidate.read_text())["status"] == "candidate"
    finally:
        root.destroy()
