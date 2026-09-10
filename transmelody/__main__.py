"""Unified entry point: python -m transmelody <command> [options]."""
import argparse
import runpy
import sys

from transmelody.paths import MODULES

CLI_MODULES = (
    "audio_folder_converter", "normalize_audio_format", "grid_initializer",
    "tempo_map", "tempo_region_grid", "midi_grid_quantizer",
    "score_transcriber_v2", "predict_test_audio", "infer_melody",
    "fusion_runtime", "prepare_dataset", "train_melody", "train_pitch_context",
    "model_release", "batch_song_renamer", "register_original_batch",
    "melody_queue_workflow", "audio_folder_converter_ui", "melody_audition_ui",
    "melody_queue_workflow_ui", "midi_quantizer_ui", "pronunciation_review_ui",
    "song_renamer_ui", "compare_melody_midi", "evaluate_checkpoint_blend",
    "evaluate_joint_decoding", "evaluate_note_artifacts",
    "evaluate_score_transcriber_v2", "evaluate_semitone_update",
    "pitch_fusion_review", "predict_fused_standalone", "lyric_alignment",
)
COMMANDS = {name: MODULES[name] for name in CLI_MODULES}
ALIASES = {
    "predict": "predict_test_audio",
    "infer": "infer_melody",
    "train": "train_melody",
    "prepare": "prepare_dataset",
    "grid": "grid_initializer",
    "tempo": "tempo_map",
    "workflow": "melody_queue_workflow",
    "ui": "melody_queue_workflow_ui",
    "convert": "audio_folder_converter",
    "quantize": "midi_grid_quantizer",
    "audition": "melody_audition_ui",
    "fusion": "fusion_runtime",
    "rename": "batch_song_renamer",
    "register": "register_original_batch",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Commands: " + ", ".join(ALIASES),
    )
    parser.add_argument("command", nargs="?", help="Command or historical script basename")
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    name = args.command.removesuffix(".py")
    name = ALIASES.get(name, name)
    if name not in COMMANDS:
        parser.error("Unknown command: " + args.command)
    sys.argv = [COMMANDS[name], *args.options]
    runpy.run_module(COMMANDS[name], run_name="__main__")


if __name__ == "__main__":
    main()
