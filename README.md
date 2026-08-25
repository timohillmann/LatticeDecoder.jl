# LatticeDecoder.jl

`LatticeDecoder.jl` provides belief-propagation and local-search decoders for
classical low-density lattice codes and GKP lattice constructions. The package
also includes constructors used by the accompanying paper: repetition,
surface, generated quantum LDLC, and bivariate-bicycle examples.

This repository is the paper-facing software release. Large-scale data
collection, plotting, scheduler scripts, and research artifacts are deliberately
kept outside the public package.

## Installation

The release supports Julia 1.11 and newer. From a clone of this repository:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

The committed manifests pin the official `LatticeAlgorithms.jl` repository to
the exact revision used for this release. `Manifest.toml` targets Julia 1.11;
Julia 1.12 automatically selects `Manifest-v1.12.toml` so both supported stdlib
sets remain reproducible.

## Quick start

The sampling API owns no global random state. Supply an explicit RNG and reuse
one mutable decoder serially:

```julia
using LatticeDecoder, Random

n, d = 16, 3
H = classical_ldlc(d, n, true)
G = generator_matrix(H)
decoder = LDLCDecoder(
    initialize_tanner_graph(H);
    schedule=:serial,
    algorithm=:lsd,
    sigma=0.35,
    max_iterations=25,
)
problem = ClassicalDecodingProblem(H, G)
estimate = estimate_symbol_error_rate!(
    MersenneTwister(2026), decoder, problem; samples=100
)

@show estimate.events estimate.trials estimate.rate
@show estimate.lower estimate.upper
```

`BinomialEstimate` retains the raw event, trial, and decoded-sample counts and
reports a clipped Agresti–Coull confidence interval. Quantum simulations use
`QuantumDecodingProblem` and `estimate_logical_error_rate!`; see the notebooks
for complete seeded examples.

## Algorithms and schedules

`LDLCDecoder` implements Gaussian-mixture belief propagation with serial and
parallel update schedules. `QuantizedLDLCDecoder` provides the quantized-density
variant. `LocalSearch` can refine integer decisions after BP. Existing decoder
and constructor names are retained in this release so paper scripts can use the
tested API directly.

## Conventions

- Generator rows are lattice checks.
- Phase-space coordinates use `qqpp` ordering: all `q` coordinates followed by
  all `p` coordinates.
- Displacements are expressed in lattice units. Multiply by `sqrt(2π)` to
  obtain the corresponding physical quadrature convention.
- A full quantum generator has determinant equal to the logical dimension in
  these units.
- Sampling routines are seeded and serial. A decoder is mutable and should not
  be shared between concurrent runs.

## Reproducible examples

Six curated notebooks cover the public workflows:

1. classical LDLC;
2. repetition CSS;
3. surface CSS;
4. surface non-CSS;
5. generated/overcomplete quantum LDLC; and
6. bivariate-bicycle CSS codes, including enlargement.

Each notebook decodes one sample and performs a small seeded sweep with
confidence bounds. Set `LATTICEDECODER_EXAMPLE_SAMPLES` to scale the sweeps;
the default is intentionally small. Fixture provenance and checksums are in
[examples/fixtures/README.md](examples/fixtures/README.md). See
[examples/00README.md](examples/00README.md) for the notebook index.

## Verification

```julia
using Pkg
Pkg.test()
```

The CI matrix tests Julia 1.11 and 1.12, runs the LSD equivalence check, executes
all notebook code cells in smoke mode, and audits the tracked release files and
fixture checksums.

## Citation

Please cite the software metadata in [`CITATION.cff`](CITATION.cff). The paper
title and DOI are marked as forthcoming and will be updated when assigned.

## License

This software is released under the [MIT License](LICENSE).
