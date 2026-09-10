"""Stable data roots and source locations for the Transmelody package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

MODULE_GROUPS = {
    "audio": (
        "audio_audition", "audio_folder_converter", "normalize_audio_format",
    ),
    "grid": (
        "grid_initializer", "tempo_map", "tempo_region_grid", "musical_timeline",
    ),
    "midi": ("midi_grid_quantizer",),
    "models": (
        "melody_transformer", "articulation_encoder", "note_event_model",
        "musical_boundary", "musical_event_decoding", "pitch_context",
        "pitch_fusion", "score_transcriber_v2",
    ),
    "inference": (
        "melody_inference", "predict_test_audio", "infer_melody",
        "joint_pitch_decoding", "fusion_runtime",
    ),
    "training": (
        "prepare_dataset", "train_melody", "train_pitch_context", "model_release",
    ),
    "workflow": (
        "batch_song_renamer", "register_original_batch", "melody_queue_workflow",
        "registry_io",
    ),
    "ui": (
        "audio_folder_converter_ui", "melody_audition_ui",
        "melody_queue_workflow_ui", "midi_quantizer_ui",
        "pronunciation_review_ui", "song_renamer_ui",
    ),
    "evaluation": (
        "melody_evaluation", "compare_melody_midi", "evaluate_checkpoint_blend",
        "evaluate_joint_decoding", "evaluate_note_artifacts",
        "evaluate_score_transcriber_v2", "evaluate_semitone_update",
        "pitch_fusion_review", "predict_fused_standalone",
    ),
    "lyrics": ("lyric_alignment", "pronunciation_labels"),
}
MODULES = {
    name: f"transmelody.{group}.{name}"
    for group, names in MODULE_GROUPS.items()
    for name in names
}
MODULES["project_settings"] = "transmelody.config"


def module_path(name):
    """Resolve a historical source filename for provenance checks."""
    key = name.removesuffix(".py")
    return PROJECT_ROOT / (MODULES[key].replace(".", "/") + ".py")
