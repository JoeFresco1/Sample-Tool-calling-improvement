# Ternary model tool-calling experiment

This repository contains the actual experiment code used to edit a local Ternary Bonsai 2 27B NInfer checkpoint for one Hermes tool-routing failure. It is a research snapshot for reading, **not a turnkey package**: the original local NInfer source/build, artifact reader, runtime, model, captured activations, experiment data and helper modules are not included. Paths and imports in these scripts reflect the original environment. No model, conversations or credentials are included.

## How the edit worked

1. `capture.cpp` wraps the compiled output-head operation and records final hidden states and logits during model generation. Actual traces showed the wrong tool name being selected inside XML function tags.
2. `fit_actual.py` selects the actual decision states, including controls for legitimate nested tool names and read calls, then invokes `fit.py`.
3. `fit.py` decodes two selected ternary output-head rows and fits a regularized least-squares correction to their per-group scales. It checks independently reconstructed logits against captured logits. The targets are token rows `write` (4775) and `tool` (13766), with 40 FP16 scales per row.
4. `edit_actual.py` creates a physical checkpoint copy, changes the 80 fitted scale words and artifact identifier, and compares the rest of the file byte-for-byte. Packed ternary codes remain unchanged. The original checkpoint is not modified. The script is experiment-specific, assumes its original model layout and fit file, and is not safe to apply to arbitrary checkpoints.

This was supervised calibration on a small set of captured examples, not full model training. The observer and production executables used different builds; decisions were checked on the sampled contexts, not every raw logit.

## Results and limitations

- Eight newly frozen, related file-writing prompts: relevant, schema-valid first tool action **0/8 → 8/8**. This does not measure completion of all eight tasks.
- Previously inspected regression cases: **83/96 → 83/96**, with no change in the aggregate score for any of the four areas tested.
- One real Hermes file/code workflow completed with each model; reported tool errors **1 → 0**. One pair is not evidence of a general speedup.
- The earlier supplied-prefix fit was rejected: 32-case score remained 27/32, and the real workflow had more errors. The later fit used actual generation states.

These small results support a narrow tool-routing improvement, not a demonstrated general improvement in coding, instruction following, or reasoning. The 96 regression cases were already inspected before this refinement. Additional tests and a fresh, broader holdout are needed for stronger claims.
