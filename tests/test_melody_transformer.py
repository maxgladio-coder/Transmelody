import pytest
import torch

from melody_transformer import (
    NUM_PITCH_CLASSES,
    NUM_RHYTHM_CLASSES,
    RHYTHM_STRAIGHT,
    RHYTHM_TRIPLET,
    STEPS_PER_BAR,
    STRAIGHT_TICKS_PER_STEP,
    TICKS_PER_STEP,
    TRIPLET_TICKS_PER_STEP,
    FeatureConfig,
    _event_labels_from_notes,
    _labels_from_notes,
    _rhythm_labels_from_notes,
    decode_rhythm_modes,
    decode_segment_events,
    pool_grid_predictions,
    quantize_pitch_classes,
    stabilize_pitch_sequence,
)


def note(start: int, end: int, pitch: int) -> dict:
    return {
        "pitch": pitch,
        "midi_time": {
            "start_tick": start,
            "end_tick": end,
        },
    }


def test_notes_become_grid_pitch_and_onset_targets() -> None:
    payload = {
        "notes": [
            note(TICKS_PER_STEP, 3 * TICKS_PER_STEP, 60),
            note(3 * TICKS_PER_STEP, 4 * TICKS_PER_STEP, 62),
        ]
    }
    pitch, onset = _labels_from_notes(payload, 5 * TICKS_PER_STEP)

    assert pitch.tolist() == [0, 61, 61, 63, 0]
    assert onset.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_polyphonic_target_is_rejected() -> None:
    payload = {
        "notes": [
            note(0, 2 * TICKS_PER_STEP, 60),
            note(TICKS_PER_STEP, 3 * TICKS_PER_STEP, 64),
        ]
    }

    with pytest.raises(ValueError, match="Polyphonic"):
        _labels_from_notes(payload, 4 * TICKS_PER_STEP)


def test_overlapping_same_pitch_notes_are_merged() -> None:
    payload = {
        "notes": [
            note(0, 2 * TICKS_PER_STEP, 60),
            note(TICKS_PER_STEP, 3 * TICKS_PER_STEP, 60),
        ]
    }

    targets = _event_labels_from_notes(payload, 4 * TICKS_PER_STEP)

    assert targets["pitch"].tolist() == [61, 61, 61, 0]
    assert targets["onset"].tolist() == [1, 0, 0, 0]
    assert targets["offset"].tolist() == [0, 0, 0, 1]
    assert targets["duration_mask"].tolist() == [True, False, False, False]


def test_target_tensors_have_expected_dtypes() -> None:
    pitch, onset = _labels_from_notes({"notes": []}, 4 * TICKS_PER_STEP)
    assert pitch.dtype == torch.long
    assert onset.dtype == torch.float32


def test_midi_notes_create_segment_boundary_and_duration_targets() -> None:
    payload = {
        "notes": [
            note(0, 3 * TICKS_PER_STEP, 60),
            note(3 * TICKS_PER_STEP, 5 * TICKS_PER_STEP, 64),
        ]
    }

    targets = _event_labels_from_notes(payload, 6 * TICKS_PER_STEP)

    assert targets["onset"].tolist() == [1, 0, 0, 1, 0, 0]
    assert targets["offset"].tolist() == [0, 0, 0, 1, 0, 1]
    assert targets["onset_pitch"].tolist() == [61, 0, 0, 65, 0, 0]
    assert targets["duration"].tolist() == [2, 0, 0, 1, 0, 0]
    assert targets["activity"].tolist() == [1, 1, 1, 1, 1, 0]
    assert targets["continuation"].tolist() == [1, 1, 0, 1, 0, 0]
    assert targets["pitch_change"].tolist() == [0, 0, 0, 1, 0, 0]


def test_midi_same_pitch_rearticulation_is_a_supervised_boundary() -> None:
    payload = {
        "notes": [
            note(0, 3 * TICKS_PER_STEP, 60),
            note(3 * TICKS_PER_STEP, 5 * TICKS_PER_STEP, 60),
        ]
    }

    targets = _event_labels_from_notes(payload, 6 * TICKS_PER_STEP)

    assert targets["articulation_boundary"].tolist() == [0, 0, 0, 1, 0, 0]
    assert targets["pitch_change"].tolist() == [0, 0, 0, 0, 0, 0]


def test_segment_decoder_ignores_pitch_fluctuation_without_onset() -> None:
    steps = 6
    pitch_logits = torch.full((steps, NUM_PITCH_CLASSES), -8.0)
    pitch_logits[:3, 61] = 10.0
    pitch_logits[1, 73] = 11.0
    pitch_logits[3:6, 65] = 12.0
    onset_logits = torch.full((steps,), -8.0)
    onset_logits[[0, 3]] = 8.0
    duration_logits = torch.full((steps, 8), -8.0)
    duration_logits[0, 2] = 8.0
    duration_logits[3, 1] = 8.0
    outputs = {
        "pitch": pitch_logits,
        "onset": onset_logits,
        "offset": torch.zeros(steps),
        "activity": torch.ones(steps),
        "duration": duration_logits,
        "continuation": torch.tensor([8.0, 8.0, -8.0, 8.0, -8.0, -8.0]),
    }

    notes = decode_segment_events(outputs)

    assert [(x.start_step, x.end_step, x.pitch) for x in notes] == [
        (0, 3, 60),
        (3, 6, 64),
    ]


def boundary_outputs(
    *,
    right_pitch: int = 60,
    articulation_at_boundary: float = -8.0,
    pitch_change_at_boundary: float = -8.0,
) -> dict[str, torch.Tensor]:
    steps = 6
    pitch_logits = torch.full((steps, NUM_PITCH_CLASSES), -8.0)
    pitch_logits[:3, 61] = 10.0
    pitch_logits[3:, right_pitch + 1] = 10.0
    onset_logits = torch.full((steps,), -8.0)
    onset_logits[[0, 3]] = 8.0
    offset_logits = torch.full((steps,), -8.0)
    offset_logits[3] = 8.0
    continuation_logits = torch.full((steps,), 8.0)
    continuation_logits[2] = -8.0
    articulation_logits = torch.full((steps,), -8.0)
    articulation_logits[3] = articulation_at_boundary
    pitch_change_logits = torch.full((steps,), -8.0)
    pitch_change_logits[3] = pitch_change_at_boundary
    return {
        "pitch": pitch_logits,
        "onset": onset_logits,
        "offset": offset_logits,
        "activity": torch.full((steps,), 8.0),
        "duration": torch.zeros((steps, 8)),
        "continuation": continuation_logits,
        "articulation": articulation_logits,
        "pitch_change": pitch_change_logits,
    }


def test_strong_same_pitch_onset_is_not_vetoed_by_articulation() -> None:
    notes = decode_segment_events(boundary_outputs())

    assert [(x.start_step, x.end_step, x.pitch) for x in notes] == [
        (0, 3, 60),
        (3, 6, 60),
    ]


def test_same_pitch_articulation_boundary_still_splits_note() -> None:
    notes = decode_segment_events(
        boundary_outputs(articulation_at_boundary=8.0)
    )

    assert [(x.start_step, x.end_step, x.pitch) for x in notes] == [
        (0, 3, 60),
        (3, 6, 60),
    ]


def test_stable_pitch_change_still_splits_note() -> None:
    notes = decode_segment_events(
        boundary_outputs(
            right_pitch=64,
            pitch_change_at_boundary=8.0,
        )
    )

    assert [(x.start_step, x.end_step, x.pitch) for x in notes] == [
        (0, 3, 60),
        (3, 6, 64),
    ]


def test_shared_grid_represents_straight_and_triplet_edges_exactly() -> None:
    payload = {
        "notes": [
            note(0, STRAIGHT_TICKS_PER_STEP, 60),
            note(
                STRAIGHT_TICKS_PER_STEP,
                STRAIGHT_TICKS_PER_STEP + TRIPLET_TICKS_PER_STEP,
                62,
            ),
        ]
    }
    pitch, onset = _labels_from_notes(
        payload,
        STRAIGHT_TICKS_PER_STEP + TRIPLET_TICKS_PER_STEP,
    )
    assert len(pitch) == 7
    assert onset.nonzero().flatten().tolist() == [0, 3]


def test_rhythm_labels_learn_straight_and_triplet_bars() -> None:
    bar_ticks = STEPS_PER_BAR * TICKS_PER_STEP
    payload = {
        "notes": [
            note(0, STRAIGHT_TICKS_PER_STEP, 60),
            note(STRAIGHT_TICKS_PER_STEP, 2 * STRAIGHT_TICKS_PER_STEP, 62),
            note(bar_ticks, bar_ticks + TRIPLET_TICKS_PER_STEP, 64),
            note(
                bar_ticks + TRIPLET_TICKS_PER_STEP,
                bar_ticks + 2 * TRIPLET_TICKS_PER_STEP,
                65,
            ),
        ]
    }
    modes, mask = _rhythm_labels_from_notes(payload, 2 * bar_ticks)
    assert int(modes[0]) == RHYTHM_STRAIGHT
    assert int(modes[STEPS_PER_BAR]) == RHYTHM_TRIPLET
    assert bool(mask[0]) and bool(mask[STEPS_PER_BAR])


def test_rhythm_labels_choose_closest_grid_for_the_whole_bar() -> None:
    bar_ticks = STEPS_PER_BAR * TICKS_PER_STEP
    payload = {
        "notes": [
            note(0, 5 * STRAIGHT_TICKS_PER_STEP, 60),
            note(
                6 * TRIPLET_TICKS_PER_STEP,
                8 * TRIPLET_TICKS_PER_STEP,
                62,
            ),
            note(
                8 * TRIPLET_TICKS_PER_STEP,
                10 * TRIPLET_TICKS_PER_STEP,
                64,
            ),
        ]
    }

    modes, mask = _rhythm_labels_from_notes(payload, bar_ticks)

    assert int(modes[0]) == RHYTHM_TRIPLET
    assert bool(mask[0])


def test_cross_bar_note_edges_vote_in_their_own_bars() -> None:
    bar_ticks = STEPS_PER_BAR * TICKS_PER_STEP
    payload = {
        "notes": [
            note(
                STRAIGHT_TICKS_PER_STEP,
                bar_ticks + TRIPLET_TICKS_PER_STEP,
                60,
            ),
        ]
    }

    modes, mask = _rhythm_labels_from_notes(payload, 2 * bar_ticks)

    assert int(modes[0]) == RHYTHM_STRAIGHT
    assert int(modes[STEPS_PER_BAR]) == RHYTHM_TRIPLET
    assert bool(mask[0]) and bool(mask[STEPS_PER_BAR])


def test_rhythm_decoder_preserves_a_confident_isolated_triplet_bar() -> None:
    logits = torch.full(
        (3 * STEPS_PER_BAR, NUM_RHYTHM_CLASSES),
        -4.0,
    )
    logits[:, RHYTHM_STRAIGHT] = 4.0
    logits[STEPS_PER_BAR : 2 * STEPS_PER_BAR, RHYTHM_TRIPLET] = 8.0
    logits[STEPS_PER_BAR : 2 * STEPS_PER_BAR, RHYTHM_STRAIGHT] = -8.0
    modes = decode_rhythm_modes(logits)
    assert modes.tolist() == [RHYTHM_STRAIGHT, RHYTHM_TRIPLET, RHYTHM_STRAIGHT]


def test_triplet_mode_rejects_onset_between_triplet_boundaries() -> None:
    steps = 9
    pitch_logits = torch.full((steps, NUM_PITCH_CLASSES), -8.0)
    pitch_logits[:, 61] = 8.0
    onset_logits = torch.full((steps,), -8.0)
    onset_logits[[0, 3, 4]] = 8.0
    outputs = {
        "pitch": pitch_logits,
        "onset": onset_logits,
        "offset": torch.zeros(steps),
        "activity": torch.ones(steps),
        "duration": torch.zeros(steps, 8),
        "continuation": torch.full((steps,), -8.0),
    }
    notes = decode_segment_events(
        outputs,
        rhythm_modes=torch.tensor([RHYTHM_TRIPLET]),
    )
    assert [note.start_step for note in notes] == [0, 4]


def test_pitch_change_head_preserves_legato_semitone_aba() -> None:
    steps = 9
    pitch_logits = torch.full((steps, NUM_PITCH_CLASSES), -8.0)
    pitch_logits[:3, 61] = 8.0
    pitch_logits[3:6, 62] = 8.0
    pitch_logits[6:, 61] = 8.0
    onset_logits = torch.full((steps,), -8.0)
    onset_logits[0] = 8.0
    pitch_change_logits = torch.full((steps,), -8.0)
    pitch_change_logits[[3, 6]] = 8.0
    outputs = {
        "pitch": pitch_logits,
        "onset": onset_logits,
        "offset": torch.zeros(steps),
        "activity": torch.ones(steps),
        "duration": torch.zeros(steps, 8),
        "continuation": torch.full((steps,), 8.0),
        "pitch_change": pitch_change_logits,
        "articulation": torch.full((steps,), -8.0),
    }

    notes = decode_segment_events(outputs)

    assert [(note.start_step, note.end_step, note.pitch) for note in notes] == [
        (0, 3, 60),
        (3, 6, 61),
        (6, 9, 60),
    ]


def test_single_cell_pitch_jitter_is_removed() -> None:
    config = FeatureConfig(n_mels=4, context_offsets=(0,))
    logits = torch.full((3, NUM_PITCH_CLASSES), -5.0)
    logits[0, 61] = 5.0
    logits[1, 63] = 5.0
    logits[2, 61] = 5.0
    onset = torch.zeros(3)
    features = torch.ones(3, config.input_dim)

    decoded, _ = stabilize_pitch_sequence(logits, onset, features, config)

    assert decoded.tolist() == [61, 61, 61]


def test_strong_onset_preserves_one_sixteenth_turn() -> None:
    config = FeatureConfig(n_mels=4, context_offsets=(0,))
    logits = torch.full((3, NUM_PITCH_CLASSES), -5.0)
    logits[0, 61] = 5.0
    logits[1, 63] = 5.0
    logits[2, 64] = 5.0
    onset = torch.tensor([0.0, 0.95, 0.95])
    features = torch.ones(3, config.input_dim)

    decoded, _ = stabilize_pitch_sequence(logits, onset, features, config)

    assert decoded.tolist() == [61, 63, 64]


def test_confident_one_cell_pitch_without_onset_is_still_jitter() -> None:
    config = FeatureConfig(n_mels=4, context_offsets=(0,))
    logits = torch.full((3, NUM_PITCH_CLASSES), -10.0)
    logits[0, 61] = 10.0
    logits[1, 72] = 10.0
    logits[2, 61] = 10.0
    onset = torch.zeros(3)
    features = torch.tensor(
        [
            [0.0, 1.0, 0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 1.0],
        ]
    )

    decoded, _ = stabilize_pitch_sequence(logits, onset, features, config)

    assert decoded.tolist() == [61, 61, 61]


def test_two_sixteenths_pool_to_one_eighth() -> None:
    logits = torch.tensor(
        [
            [2.0, 4.0],
            [4.0, 2.0],
            [1.0, 5.0],
            [1.0, 3.0],
        ]
    )
    onset = torch.tensor([0.1, 0.8, 0.2, 0.3])
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    pooled_logits, pooled_onset, pooled_features = pool_grid_predictions(
        logits,
        onset,
        features,
        factor=2,
    )

    assert pooled_logits.tolist() == [[3.0, 3.0], [1.0, 4.0]]
    assert torch.allclose(pooled_onset, torch.tensor([0.8, 0.3]))
    assert pooled_features.tolist() == [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]


def test_eighth_quantization_never_invents_pitch() -> None:
    pitch = torch.tensor([61, 63, 64, 64])
    logits = torch.zeros(4, NUM_PITCH_CLASSES)
    logits[0, 61] = 4.0
    logits[1, 63] = 5.0
    logits[2:, 64] = 4.0
    # An unrelated class has a moderately high score in both halves. Averaging
    # logits could select it, but hard-class quantization must not.
    logits[:2, 70] = 4.8
    onset = torch.tensor([0.1, 0.9, 0.2, 0.3])

    quantized, quantized_onset = quantize_pitch_classes(
        pitch,
        logits,
        onset,
        factor=2,
    )

    assert int(quantized[0]) in {61, 63}
    assert quantized.tolist() == [63, 64]
    assert torch.allclose(quantized_onset, torch.tensor([0.9, 0.3]))
