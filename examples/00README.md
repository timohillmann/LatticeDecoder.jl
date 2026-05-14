# Examples

This directory is for small, single-purpose examples that demonstrate how to use
the public `LatticeDecoder` API.

Examples should:

- run from a fresh package environment with `julia --project=. examples/<file>.jl`;
- avoid distributed jobs, plotting, and large data files unless the example is
  specifically about those features;
- use `using LatticeDecoder` instead of including files from `src/` directly.

Long simulation runs, figure-generation scripts, and data collection workflows
belong in a separate experiment or generated-data directory.

## Classical LDLC decoding

Open:

```sh
examples_new/classical_ldlc_decode.ipynb
```

The example constructs a small classical LDLC code, encodes a random binary
message, adds Gaussian noise, runs belief propagation, and reports the number of
symbol errors after hard decision decoding.

## CSS surface-code decoding

Open:

```sh
examples_new/surface_code_css_decode.ipynb
```

The example constructs a small GKP surface code, selects one CSS sector, runs
serial belief propagation on a single Gaussian displacement, checks whether the
residual is logically trivial, and includes a small `d = 3, 5, 7` sweep over six
noise values.

## CSS repetition-code decoding

Open:

```sh
examples_new/rep_code_css_decode.ipynb
```

The example constructs a small GKP repetition code, selects one CSS sector, runs
serial belief propagation on a single Gaussian displacement, checks whether the
residual is logically trivial, and includes a small `d = 3, 5, 7` sweep.

## Non-CSS surface-code decoding

Open:

```sh
examples_new/surface_code_noncss_decode.ipynb
```

The example constructs the full GKP surface-code lattice, decodes a single
coupled `q`/`p` displacement, and checks whether the residual is logically
trivial.

## Generated qLDLC decoding

Open:

```sh
examples_new/qldlc_generated_code_decode.ipynb
```

The example loads a generated qLDLC code by name, prepares the decoding
experiment, and runs a small smoke decode.

## Bivariate bicycle code enlargement

Open:

```sh
examples/bivariate_bicycle_enlarge.ipynb
```

The example loads every non-expanded `.npz` file in
`generator_matrices/bivariate_bicycle`, enlarges `hx` and `hz` into full-rank
lattice generators, and writes metadata-preserving files with `Mqq` and `Mpp`
under `generator_matrices/bivariate_bicycle/expanded`. Weight balancing is
enabled when loading the source matrices and can be toggled with the
`balance_weights` variable.

## Expanded bivariate bicycle CSS decoding

Open:

```sh
examples/bivariate_bicycle_css_decode.ipynb
```

The example ensures expanded BB files exist, selects one CSS sector, runs
belief propagation on a single Gaussian displacement, includes opt-in local
search, and runs a tiny smoke sweep over the expanded BB codes.
