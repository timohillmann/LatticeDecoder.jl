module LatticeDecoder

export stack_gkp_generator
include("lattice_tools/concatenated_code_reduction.jl")
# Code code_constructors
export GkpRepCode, QuantumCode, gkp_rep_code, rep_code_logical, is_logical_error, GKP_Rep_Code
include("code_constructors/rep_codes.jl")

export GKP_Surface_Code
include("code_constructors/surface_codes.jl")

# export lattice_statistics_decoding, LatticeStatisticsDecoding, lattice_statistics_decoding!, local_search!
# include("lattice_osd/lattice_statistics_decoding.jl")

export initialize_tanner_graph_quant, run_belief_propagation!
include("bp_algorithms/quantized_decoder_parallel.jl")

export LocalSearch, local_search!
include("lattice_osd/local_search.jl")

# BP tools
export initialize_tanner_graph, run_belief_propagation!
# include("bp_algorithms/parallel_bp.jl")
include("bp_algorithms/parallel_bp_log_weight.jl")

export run_serial_belief_propagation!
include("bp_algorithms/serial_bp_log_weight.jl")

# LDLC tools
export classical_ldlc, generator_matrix, encode, decode, encode!, decode!
include("code_constructors/classical_ldlc.jl")

# Utilities
export sample_error, hard_decision, count_symbol_errors, lattice_capacity_var, lattice_capacity_std, snr, snr_db, symplectic_form
include("utilities/utilities.jl")

export metadata, add_data!
include("utilities/tools.jl")


export overcomplete_syndrome_preperation, integer_solve, compute_syndrome, compute_eta, compute_eta_overcomplete
include("lattice_tools/overcomplete_syndrome.jl")

# export l1_minimize
# include("lp_decoding/lp_decoding.jl")

end # module LatticeDecoder
