# Transmelody v0.1.0 model card

## Intended use

Human-in-the-loop lead-melody score drafting from already separated vocal and
instrumental stems, primarily Japanese pop songs in 4/4. Not all-track transcription,
not a source separator, not a chord model, and not reliable unattended notation.

## Architecture

The base network uses 96-bin log-mel features with 13 local frames, a validity
indicator, and 771 frozen speech features, giving 2020 input channels. It uses a
128-wide, four-layer, four-head Transformer plus supervised event/context modules.
MIDI targets supervise pitch, activity, onset, offset, duration, continuation and
related event signals. A complete-note decoder constructs the output spans.

The auxiliary network takes continuous 360-bin RMVPE salience and 128-bin mel
features, with local acoustic attention and a three-layer Transformer. At inference,
log-softmax pitch scores are pooled over each base note span and combined with
weight 0.25 for base and 0.75 for auxiliary. This is a weighted score ensemble,
not a calibrated confidence probability and not two models voting on segmentation.

Queries use a shared 40-tick lattice (PPQ 480). Production decoding selects
straight sixteenth or eighth-triplet modes by bar. Tempo-map MIDI and feature
sampling share one timeline with an explicit WAV START offset and full-bar padding.

## Development and evidence

The source project had 19 annotated songs, approximately 1.44 hours including
instrumental sections and 9,514 reference notes. Base development used a
song-level 16/3 split. The same three development songs were repeatedly consulted
during model and decoder changes; they are **not a fresh test set**.

On that reused development set, pitch fusion changed full-note F1 from 0.7094 to
0.7328, and pitch+onset F1 from 0.8148 to 0.8448. Full-note matching and other
metric definitions are in `melody_evaluation.py`. These are internal comparisons,
not evidence of general accuracy or superiority over commercial tools.
One development song's semitone ABA recovery worsened from 4/17 to 3/17, despite
an aggregate improvement. Fusion was selected provisionally after listening.

The publicly provided weights preserve source model tensors exactly. Metadata was
sanitized; exact optimizer resumption and private dataset reproduction are not
provided. The ordinary workflow trains only the base, so updated model pairs
need fresh evaluation. Do not classify a potentially previously-seen song as an
unseen test merely because release provenance omits its private identifier.

## Limits and risks

Timing mistakes can dominate perceived model quality. The current automatic grid
may convert imperfect or interpolated beat anchors into spurious tempo sections,
especially around syncopated drums and weak endings. User-confirmed constant BPM
overrides are supported; a general confidence-aware tempo detector remains work
to do. Real variable-tempo material is supported through the shared tempo timeline,
but automatic detection is not guaranteed.

Fusion cannot repair merged or fragmented base note spans. Breath/noise,
background harmony, delay, semitone turns, variable pronunciation and language
shift remain failure cases. Long-note and short-note errors both occur. Frozen
Japanese ASR representations do not constitute verified kana, lyrics or syllable
boundaries. Audio and MIDI must be aligned correctly before training.

Source songs and annotations are not distributed. The source-code MIT license
does not grant rights to music, recordings, lyrics, third-party weights or voices.
