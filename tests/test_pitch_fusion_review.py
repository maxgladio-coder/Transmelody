import pytest
import torch

from transmelody.models.melody_transformer import PredictedNote
from transmelody.evaluation.pitch_fusion_review import fuse_pitches, eligible


def logits(pitches):
    values = torch.full((len(pitches),129),-5.)
    for i,p in enumerate(pitches):
        values[i,p+1] = 5
    return values


def test_zero_weight_preserves_production_exactly():
    notes = [PredictedNote(0,3,65,.9,3)]
    assert fuse_pitches(notes,logits([60]*3),logits([61]*3),0) == notes


def test_new_pitch_cannot_add_splits_or_fill_rests():
    notes = [PredictedNote(0,12,60,.8,12),PredictedNote(15,18,61,.9,3)]
    out = fuse_pitches(notes,logits([60]*18),logits([62]*6+[63]*3+[62]*6+[62]*3),1)
    assert [(n.start_step,n.end_step,n.pitch,n.onset_probability,n.duration_steps) for n in out] == [
        (0,12,62,.8,12),(15,18,62,.9,3)]
    assert notes[0].pitch == 60


def test_adjacent_same_pitch_notes_remain_separate():
    notes = [PredictedNote(0,3,60,1.,3),PredictedNote(3,6,61,1.,3)]
    out = fuse_pitches(notes,logits([60]*6),logits([62]*6),1)
    assert len(out)==2 and out[0].end_step==out[1].start_step and all(n.pitch==62 for n in out)


def test_real_semitone_aba_not_suppressed():
    notes = [PredictedNote(i,i+3,60,1.,3) for i in (0,3,6)]
    out = fuse_pitches(notes,logits([60]*9),logits([60]*3+[61]*3+[60]*3),.75)
    assert [n.pitch for n in out]==[60,61,60]


@pytest.mark.parametrize('weight',[-.1,1.1,float('nan')])
def test_bad_weight_rejected(weight):
    with pytest.raises(ValueError):
        fuse_pitches([],logits([60]),logits([60]),weight)


def test_misaligned_logits_rejected():
    with pytest.raises(ValueError,match='timeline'):
        fuse_pitches([],logits([60]),logits([60,61]),.5)


def test_aggregate_gain_does_not_hide_one_song_aba_regression():
    baseline = {'development': {'full_note_f1': .7, 'pitch_onset_f1': .8},
        'songs': {'18': {'aba_recovered': 4, 'short_note_matches': 169}}}
    candidate = {'development': {'full_note_f1': .74, 'pitch_onset_f1': .84},
        'songs': {'18': {'aba_recovered': 3, 'short_note_matches': 175}}}
    assert not eligible(candidate,baseline,['18'])
