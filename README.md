# Decoding tools for Lattice Codes

## Repo Structure
1. `src/` contains the source code for the decoding algorithms.
    1. `bp_algorithms/` contains the source code for the belief propagation decoding algorithms.
        1. `parallel_bp.jl` Contains message passing functions for the parallel belief propagation algorithm.
        2. `serial_bp.jl` Contains message passing functions for the serial belief propagation algorithm.
        3. `tanner_graph.jl` Contains functions for creating the Tanner graph of a code.
        4. `gaussian.jl` Defines the `gaussian` struct and related operations on gaussian types.
        5. `gaussian_log_weights.jl` Defines the `gaussian_log_weight` struct and related operations on gaussian log weight types.
        6. `list_sphere_decoder.jl` Contains the source code for the list sphere decoder and variable node update rules.
    2. `code_constructors/` contains the source code for constructing lattice codes.
        1. `


## Conventions
- Quadrature ordering is `qqpp`

- We stick to Lattice units in the sense that the full generator matrix `M_tot`, containing q and p sections, has determinant equal to the logical dimension.
- Rows of the generator matrix are checks of the code.
- The first `n` columns of the generator matrix descripe to `q` sector of the lattice code. The next `n` columns describe the `p` sector of the lattice code.
- We would like to allow for non-CSS codes and thus logical operators of length `2n`
- `M_q` and `M_p` correspond to `H_X` and `H_Z` respectively, in CSS codes.
- $D(\vec{\xi}) = e^{-(\vec{\xi}^T J \vec{x}) \sqrt{2 \pi } i }$ which means displacements are expressed in lattice units.