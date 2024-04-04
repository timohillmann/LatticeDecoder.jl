module LatticeDecoder

# Code code_constructors
export GkpRepCode, QuantumCode, gkp_rep_code, rep_code_logical, is_logical_error, GKP_Rep_Code
include("code_constructors/rep_codes.jl")

# BP tools
export initialize_tanner_graph, run_belief_propagation!
# include("bp_algorithms/parallel_bp.jl")
include("bp_algorithms/parallel_bp_log_weight.jl")

# LDLC tools
export classical_ldlc, generator_matrix, encode, decode, encode!, decode!
include("code_constructors/classical_ldlc.jl")

# Utilities
export sample_error, hard_decision, count_symbol_errors, lattice_capacity_var, lattice_capacity_std, snr, snr_db, symplectic_form
include("utilities/utilities.jl")

export overcomplete_syndrome_preperation, integer_solve, compute_syndrome, compute_eta, compute_eta_overcomplete
include("lattice_tools/overcomplete_syndrome.jl")

end # module LatticeDecoder
