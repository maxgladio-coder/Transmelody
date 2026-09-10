import copy
import json
import sys

import numpy as np
import pytest
import torch

from lyric_alignment import group_phones, validate_phrases, write_new, main
from pronunciation_labels import SCHEMA, digest, load_reviewed, reviewed_targets, validate_labels


def labels():
    return {"schema_version": SCHEMA, "time_reference": "original_wav_seconds",
        "status": "reviewed", "reviewer": "test", "song_id": "1", "audio_sha256": "example",
        "audio_duration": 3., "reviewed_regions": [[0., 1.]],
        "units": [{"start": .2, "end": .8, "label": "k a"}],
        "phones": [{"start": .2, "end": .3, "label": "k"}, {"start": .3, "end": .8, "label": "a"}]}


def test_cv_grouping_retains_repetitions_and_normalizes_devoiced_vowels():
    assert group_phones("k a k a sh I N cl t o".split()) == [["k", "a"], ["k", "a"], ["sh", "i"], ["N"], ["cl"], ["t", "o"]]
    with pytest.raises(ValueError):
        group_phones(["k", "pau", "a"])


@pytest.mark.parametrize("change", ["candidate", "no_regions", "overlap", "nan", "bad_time_reference"])
def test_reject_untrusted_and_invalid_labels(change):
    data = labels()
    if change == "candidate":
        data["status"] = "candidate"
    elif change == "no_regions":
        data["reviewed_regions"] = []
    elif change == "overlap":
        data["units"].append({"start": .7, "end": .9, "label": "i"})
    elif change == "nan":
        data["units"][0]["end"] = float("nan")
    else:
        data["time_reference"] = "midi_seconds"
    with pytest.raises(ValueError):
        validate_labels(data, require_reviewed=True)


def test_phone_internal_edge_not_a_unit_boundary_and_unreviewed_is_unknown():
    times = np.arange(0., 3., .01)
    target, mask = reviewed_targets(labels(), times)
    assert target[20] == 1 and target[80] == 1
    assert target[30] < 1e-4  # /k/ -> /a/ is NOT another lyric unit
    assert not mask[100:].any()
    assert not mask[0] and mask[20]


def test_between_grid_points_edge_is_not_lost():
    target, mask = reviewed_targets(labels(), np.array([.025, .075, .125, .175, .225, .275]))
    assert target[3] > .60 and target[4] > .60
    assert mask.all()


def test_source_hash_and_song_id_checked(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original")
    data = labels()
    data["audio_sha256"] = digest(audio)
    path = tmp_path / "1.json"
    write_new(path, data)
    assert load_reviewed(path, audio, "1")["reviewer"] == "test"
    with pytest.raises(ValueError):
        load_reviewed(path, audio, "2")
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_reviewed(path, audio, "1")
    with pytest.raises(FileExistsError):
        write_new(path, data)


def test_timed_clips_validate_and_keep_repeated_lines():
    phrases = [{"text": "repeat", "start": 2., "end": 4.}, {"text": "repeat", "start": 4., "end": 7.}]
    assert validate_phrases(phrases, 10.)
    phrases[1]["start"] = 3.
    with pytest.raises(ValueError):
        validate_phrases(phrases, 10.)
    assert not validate_phrases([{"text": "repeat"}, {"text": "repeat"}], 10.)


def test_accept_requires_explicit_review_and_never_overwrites(tmp_path, monkeypatch):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"original")
    data = labels()
    data.update(status="candidate", reviewed_regions=[], audio_sha256=digest(audio))
    candidate = tmp_path / "candidate.json"
    write_new(candidate, data)
    monkeypatch.setattr(sys, "argv", ["lyric_alignment.py", "accept", str(candidate), str(audio),
        "--reviewer", "Human", "--dataset", str(tmp_path)])
    with pytest.raises(ValueError):
        main()
    data["reviewed_regions"] = [[0., 1.]]
    candidate.write_text(json.dumps(data))
    main()
    saved = load_reviewed(tmp_path / "pronunciation_labels/1.json", audio, "1")
    assert saved["review_scope"] == "pronunciation_units"
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize("with_unknown", [False, True])
def test_coarse_ctc_keeps_repeated_tokens_and_phrase_order(monkeypatch, with_unknown):
    import types
    import transformers
    import articulation_encoder
    from lyric_alignment import coarse_phrase_times
    class Encoder:
        config = types.SimpleNamespace(pad_token_id=0, conv_kernel=[1], conv_stride=[160])
        def __call__(self, waveform):
            logits = torch.full((1, 100, 3), -10.)
            logits[:, :, 0] = 10
            for a, b, token in ((10, 15, 1), (20, 25, 1), (50, 55, 1), (70, 75, 1)):
                logits[:, a:b, 0] = -10
                logits[:, a:b, token] = 10
            return types.SimpleNamespace(logits=logits)
    # Exercise the real model's shared blank/UNK ID without dropping its slot.
    tokens = [1, 1, 1, 0, 1] if with_unknown else [1, 1]
    tokenizer = types.SimpleNamespace(unk_token_id=0 if with_unknown else 2, encode=lambda text, **_: tokens)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: tokenizer)
    monkeypatch.setattr(articulation_encoder, "_load_encoder", lambda *_: Encoder())
    monkeypatch.setattr(articulation_encoder, "clear_articulation_encoder_cache", lambda: None)
    result = coarse_phrase_times(np.zeros(16000, dtype=np.float32), 16000,
        [{"text": "aa"}, {"text": "aa"}], torch.device("cpu"))
    assert len(result) == 2
    assert result[0]["start"] < result[0]["end"] <= result[1]["start"] < result[1]["end"]
    assert result[0]["coarse_unknown_tokens"] == int(with_unknown)


def test_review_region_union():
    from pronunciation_review_ui import merge_regions
    assert merge_regions([[3, 4], [0, 1], [1, 2]]) == [[0, 2], [3, 4]]


def test_boundary_only_export_does_not_reproduce_source_text():
    from lyric_alignment import boundaries_only
    data = labels()
    data["lyrics"] = {"phrases": [{"text": "source words"}]}
    data["phrases"] = [{"text": "source words", "reading": "source reading", "units": [["s", "o"]], "start": 0., "end": 1.}]
    result = boundaries_only(data, "https://example.org/source")
    assert "lyrics" not in result and result["phrases"][0]["text"] == "Phrase 01"
    assert result["units"][0]["label"] == "unit_0001"
    assert result["phones"][0]["label"] == "phone_0001"
    assert data["units"][0]["label"] == "k a"


def test_diagnostic_flags_do_not_rewrite_short_units():
    from lyric_alignment import candidate_diagnostics
    data = labels()
    data["phrases"] = [{"start": 0., "end": 1., "sofa_confidence": .5, "coarse_unknown_tokens": 1}]
    data["units"][0].update(start=.2, end=.21, phrase=0)
    previous = copy.deepcopy(data)
    report = candidate_diagnostics(data)
    assert report["under_20ms_count"] == 1 and report["coarse_unknown_phrase_numbers"] == [1]
    assert data == previous
