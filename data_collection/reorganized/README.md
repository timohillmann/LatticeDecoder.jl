# Decode Reorganized Generated Codes

This folder contains a decoder runner for reduced codes generated in:

- `/Users/timo/Documents/gkp_ldlc_mwe-main/examples/reorganized/`
- `/Users/timo/Documents/gkp_ldlc_mwe-main/examples/reduced_ldlc_gens/`

The script keeps the same simulation data saving structure used in the existing data collection scripts (`add_data!` with `json_metadata` in CSV).

## Script

- `decode_generated_codes.jl`
- `decode_fra_codes.jl`
- `plot_generated_codes.py`

What it does:

1. Finds `.jls` reduced-code artifacts.
2. Resolves `reduced_generator_path` and `reduced_generator_inv_path` (with filename fallback).
3. Loads those `.npy` matrices as `G` and `H`.
4. Runs decoding over configured sigma values.
5. Appends rows to a results CSV in the same format as other scripts in this repo.

## Quick Start

Run from repo root:

```bash
julia --project=. data_collection/reorganized/decode_generated_codes.jl
```

This script is configured from inside the file itself via `RUN_SETTINGS`:

- [decode_generated_codes.jl](/Users/timo/Documents/LatticeDecoder.jl/data_collection/reorganized/decode_generated_codes.jl)

## Output

Default output file:

- `results/reorganized_codes/reduced_ldlc_generated_decoding.csv`

Each row is written via `add_data!` and includes:

- `shots`
- `errors`
- `decoder`
- `json_metadata`

The metadata includes the usual decoding parameters plus per-code provenance:

- `code_name`, `n`, `task_id`, `attempt`, `code_index`
- `code_file`
- `reduced_generator_path`
- `reduced_generator_inv_path`
- `matrix_inverse_residual`

## Script Configuration

Edit `RUN_SETTINGS` near the bottom of the script to control runs.

Main fields:

- `source_root`, `reduced_codes_dir`, `results_path`
- `start_code_index`, `end_code_index`, `max_codes`, `n_samples`, `repeats`, `target_workers`
- `sigmas`, `sigmas_already_normalized`
- `decoder`, `schedule`, `decoding_style`
- `search_radius`, `local_search`, `local_search_lll`, `sphere_decoding`, `full_basis`

`start_code_index`/`end_code_index` are 1-based indices over the script's sorted artifact list.
Use these to resume from a later point, e.g. start at code `25`.

`sigmas` are interpreted as physical values by default and normalized by `sqrt(2π)` internally.  
Set `sigmas_already_normalized = true` if you already provide normalized sigma values.

## Example Script-First Setup

Inside `RUN_SETTINGS`:

```julia
max_codes = 1,
n_samples = 10,
repeats = 1,
target_workers = 1,
sigmas = [0.35],
results_path = "results/reorganized_codes/reduced_ldlc_generated_decoding_smoke.csv",
```

Then run:

```bash
julia --project=. data_collection/reorganized/decode_generated_codes.jl
```

## Plotting

The plotting script is modeled after the existing quantum LDLc plotting scripts and targets:

- `results/reorganized_codes/reduced_ldlc_generated_decoding.csv`

Run:

```bash
python3 data_collection/reorganized/plot_generated_codes.py
```

It generates plots in:

- `results/figures/reorganized_codes/`

Edit the top-level settings in `plot_generated_codes.py` (`DECODER`, `SCHEDULE`, `INCLUDE_NS`, `SCALING_SIGMAS`, etc.) to control filtering and figure variants.

## FRA `.jld2` Codes

The FRA loader/runner handles files in:

- `data/generator_matrices/ldlc/reduced_ldlc`

Run a FRA decode batch from repo root:

```bash
julia --project=. data_collection/reorganized/decode_fra_codes.jl
```

`decode_fra_codes.jl` reuses the sampling, decoding, and CSV writing code from `decode_generated_codes.jl`, but discovers `.jld2` FRA artifacts instead of `.jls`/`.npy` generated artifacts. It decodes with the `qubit_generator_lll` matrix and computes `H = inv(G)` internally.

Default output file:

- `results/reorganized_codes/fra_ldlc_decoding.csv`
