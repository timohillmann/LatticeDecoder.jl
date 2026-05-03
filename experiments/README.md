# Experiments

This directory contains scripts and helpers for larger decoding experiments.
The package code lives in `src/`, while small public API demonstrations live in
`examples/`.

## Layout

- `QuantumDecodingExperiments.jl` loads the experiment helper module.
- `shared.jl` contains reusable sampling, sweep, metadata, and logical-error code.
- `codes/` contains code-family-specific problem builders.
- `code_generation/` contains exploratory code search, expansion, distance-checking,
  and conversion scripts used to prepare code families or generated data.
- `decode_qldlc_codes.jl` decodes generated qLDLC JLD2 code files from
  `generated_qLDLCs/`.
- `runs/` contains executable experiment scripts.

Run scripts from the repository root, for example:

```sh
julia --project=. experiments/runs/surface_code_css.jl
```
