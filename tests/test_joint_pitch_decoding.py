import torch
import pytest

from transmelody.inference.joint_pitch_decoding import decode_pitch_boundary_sequence
from transmelody.models.musical_boundary import HOLD, START, END, REST


def fixture(length=48):
    out = {'pitch': torch.full((length, 129), -8.), 'boundary_state': torch.full((length, 4), -8.)}
    out['pitch'][:, 61] = 8.
    out['boundary_state'][:, HOLD] = 8.
    out['boundary_state'][0, START] = 20.
    return out


def decode(out, modes=None, observed=None):
    n = len(out['pitch'])
    return decode_pitch_boundary_sequence(out, torch.zeros((n + 47)//48, dtype=torch.long) if modes is None else modes,
        torch.ones(n, dtype=torch.bool) if observed is None else observed)


def test_stable_pitch_recovers_missed_turn_despite_local_hold():
    out = fixture()
    out['pitch'][12:24, 61] = -8
    out['pitch'][12:24, 62] = 8
    # Local HOLD wins, but its fixed-pitch explanation fails sustained evidence.
    notes = decode(out)
    assert [(n.start_step, n.end_step, n.pitch) for n in notes] == [(0, 12, 60), (12, 24, 61), (24, 48, 60)]


def test_one_frame_pitch_wobble_does_not_force_split():
    out = fixture()
    out['pitch'][12, 61] = -1
    out['pitch'][12, 62] = 1
    notes = decode(out)
    assert [(n.start_step, n.end_step, n.pitch) for n in notes] == [(0, 48, 60)]


def test_same_pitch_rearticulations_survive():
    out = fixture(12)
    out['boundary_state'][[3, 6, 9], START] = 20.
    assert [n.start_step for n in decode(out)] == [0, 3, 6, 9]


def test_straight_start_triplet_end_and_silence():
    out = fixture(96)
    out['pitch'].fill_(-8)
    out['pitch'][:, 0] = 8
    out['pitch'][45:52, 0] = -8
    out['pitch'][45:52, 68] = 8
    out['boundary_state'].fill_(-8)
    out['boundary_state'][:, REST] = 8
    out['boundary_state'][45, START] = 20
    out['boundary_state'][48, HOLD] = 20
    out['boundary_state'][52, END] = 20
    notes = decode(out, torch.tensor([0, 1]))
    assert [(n.start_step, n.end_step, n.pitch) for n in notes] == [(45, 52, 67)]


def test_observation_gap_cannot_be_bridged_or_generate_virtual_notes():
    out = fixture(12)
    valid = torch.ones(12, dtype=torch.bool)
    valid[5:] = False
    notes = decode(out, observed=valid)
    assert len(notes) == 1 and notes[0].end_step == 5
    assert not decode(out, observed=torch.zeros_like(valid))


def test_requires_boundary_head_and_valid_weight():
    with pytest.raises(ValueError):
        decode_pitch_boundary_sequence({}, torch.tensor([0]), torch.ones(12, dtype=torch.bool))
    with pytest.raises(ValueError):
        decode_pitch_boundary_sequence(fixture(), torch.tensor([0]), torch.ones(48, dtype=torch.bool), pitch_weight=-1)
