# Reproducible notebooks

The six notebooks are intentionally small, seeded demonstrations of the public
API. They contain no stored execution output and use only files committed to
this repository.

| Notebook | Workflow |
| --- | --- |
| `classical_ldlc_decode.ipynb` | Classical LDLC symbol-error sampling |
| `rep_code_css_decode.ipynb` | Repetition CSS logical decoding and local search |
| `surface_code_css_decode.ipynb` | CSS surface-code construction and decoding |
| `surface_code_noncss_decode.ipynb` | Non-CSS surface-code decoding |
| `qldlc_generated_code_decode.ipynb` | Generated and overcomplete qLDLC decoding |
| `bivariate_bicycle_css_decode.ipynb` | Bivariate-bicycle decoding and enlargement |

Every notebook includes one decoded sample, a seeded sweep, an Agresti–Coull
confidence interval, and a note explaining how to increase the run size. The
default sweep uses 20 samples. For a fast smoke run:

```sh
LATTICEDECODER_EXAMPLE_SAMPLES=1 julia --project=. test/execute_notebooks.jl
```

For larger paper-style local runs, set the same variable to the desired sample
count. Run independent seeds as separate serial processes; do not share a
mutable decoder across workers.

Minimal input arrays are documented in [`fixtures/README.md`](fixtures/README.md).
