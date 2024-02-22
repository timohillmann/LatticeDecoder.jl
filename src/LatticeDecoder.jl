module LatticeDecoder

using LatticeDecoder

# BP tools
export initialize_tanner_graph, run_belief_propagation!
include("bp_algorithms/tanner_graph.jl")
include("bp_algorithms/parallel_bp.jl")

# LDLC tools
export classical_ldlc
include("code_constructors/classical_ldlc.jl")


# Utilities
export sample_error, hard_decision, count_bit_errors
export ec_experiment, lattice_capacity_var, lattice_capacity_std, snr, snr_db
include("utiilities/utilities.jl")
include("utiilities/simulation.jl")


end # module LatticeDecoder
