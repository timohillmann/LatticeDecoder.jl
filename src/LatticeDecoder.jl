module LatticeDecoder

# using LatticeDecoder

# BP tools
include("bp_algorithms/tanner_graph.jl")
include("bp_algorithms/parallel_bp.jl")
include("bp_algorithms/serial_bp.jl")
export initialize_tanner_graph, run_belief_propagation!, run_serial_belief_propagation!

# LDLC tools
include("code_constructors/classical_ldlc.jl")
export classical_ldlc, load_ldlc


# Utilities
include("utiilities/utilities.jl")
export sample_error, hard_decision, count_symbol_errors, lattice_capacity_var, lattice_capacity_std, snr, snr_db

end # module LatticeDecoder
