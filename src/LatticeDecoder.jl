module LatticeDecoder


# BP tools
export initialize_tanner_graph, run_belief_propagation!
# include("bp_algorithms/parallel_bp.jl")
include("bp_algorithms/parallel_bp_log_weight.jl")

# LDLC tools
export classical_ldlc, generator_matrix, encode, decode, encode!, decode!
include("code_constructors/classical_ldlc.jl")


# Utilities
export sample_error, hard_decision, count_symbol_errors, lattice_capacity_var, lattice_capacity_std, snr, snr_db
include("utilities/utilities.jl")

end # module LatticeDecoder
