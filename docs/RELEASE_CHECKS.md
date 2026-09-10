# Release checks

## v0.1.1: functional package layout

- Source moved into the `transmelody` package by responsibility; no flat root-level
  business Python modules remain. Existing Windows GUI launchers are preserved.
- New `python -m transmelody` entry point, qualified internal imports and workflow
  subprocess commands. Data roots remain unchanged through `transmelody.paths`.
- Working-project suite: 178 passed, 4 optional tests skipped. Public-release suite:
  181 passed, 4 optional tests skipped. New tests cover root paths, command aliases,
  source provenance resolution, option forwarding and GUI/workflow launch paths.
- Re-ran real-song prediction through the new CLI: the 406-note MIDI and tempo map
  remained byte-identical to the pre-reorganization working outputs.
- Learned checkpoint tensors, source audio, reference MIDI and queue state were
  not changed. The v0.1.0 model assets remain the ones used by this code release.

## v0.1.0: original model package

- Portable release suite: 174 passed, 4 optional external-model tests skipped.
- Base and auxiliary exported checkpoint state tensors: exact equality with the
  working mainline checkpoints. File hashes differ because private metadata was removed.
- Real-song smoke test: regenerated a 254.6-second, 44.1 kHz stereo stem pair at
  user-confirmed 176 BPM. Both the 406-note melody MIDI and the tempo-only MIDI
  were byte-identical to the working mainline outputs (PPQ 480, 188 padded bars).
- Smoke audio, reference data and predicted MIDI are excluded from Git and archives.
- Registry storage is replaced with openpyxl in this distribution; an empty registry
  is initialized on setup. The user's existing working registry was not changed.
- Fusion policy base path is relative to its policy file, with asset checksum checks.
- This smoke test verifies packaging parity, not transcription accuracy. It reuses
  the known grid and local third-party model cache; first-time downloads and other
  operating systems require their own environment checks.
