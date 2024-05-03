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
        1. `rep_codes.jl` Contains functions for constructing repetition codes.
        2. `surface_codes.jl` Contains functions for constructing toric codes.
        3. `classical_ldlc.jl` Contains functions for constructing classical low density lattice codes.
        4. `quantum_ldlc.jl` Contains functions for constructing quantum low density lattice codes.
    3. `lattice_tools/` contains the source code for lattice operations. _These things can be imported from LatticeAlgorithms.jl_
        1. `svp.jl` Contains functions for solving the shortest vector problem.
        2. `cvp.jl` Contains functions for solving the closest vector problem.
        3. `canonize_lattice.jl` Contains functions for canonanizing lattices.
        4. `concatenated_code_reduction.jl` greedy algorithm to reduce overcomplete set to lattce basis for a lattice code
    4. `post_processing/` contains the source code for post processing the results of the decoding algorithms. Might contain function for local search around the BP solution, calls to OSD for optimizing binary check solution etc.
        1. 
2. `examples/` contains example code for using the decoding algorithms and how to perform large scale simulation runs.
    1. `binary_code_to_qec_simulation.ipynb` contains an example of how to use the decoding algorithms to simulate a the rep code, includiing construction of the lattice generator matrix for different distances, initiliazing the Tannger graph and running the BP algorithm on it, both on th reduced basis and the overcomplete basis.
    2. `classical_ldlc_simulation.ipynb` contains an example of how to use the decoding algorithms to simulate a the classical low density lattice code, includiing construction of the lattice generator matrix for different distances, initiliazing the Tannger graph and running the BP algorithm on it, both on th reduced basis and the overcomplete basis.

## Conventions
- Quadrature ordering is `qqpp`

- We stick to Lattice units in the sense that the full generator matrix `M_tot`, containing q and p sections, has determinant equal to the logical dimension.
- Rows of the generator matrix are checks of the code.
- The first `n` columns of the generator matrix descripe to `q` sector of the lattice code. The next `n` columns describe the `p` sector of the lattice code.
- We would like to allow for non-CSS codes and thus logical operators of length `2n`
- `M_q` and `M_p` correspond to `H_X` and `H_Z` respectively, in CSS codes.
- $D(\vec{\xi}) = e^{-(\vec{\xi}^T J \vec{x}) \sqrt{2 \pi } i }$ which means displacements are expressed in lattice units. So physical units are obtained by multiplying by $\sqrt{2 \pi}$.

## Useful Resources
- For easy access to (quantum) codes, we can use the 'CodingTheory' package in Julia.
