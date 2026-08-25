module LatticeDecoder

include("constants.jl")

# Code code_constructors
export stack_gkp_generator
include("code_constructors/gkp_generators.jl")

export QuantumCode, gkp_rep_code, rep_code_logical, is_logical_error, GKP_Rep_Code
include("code_constructors/rep_codes.jl")

export GKP_Surface_Code
include("code_constructors/surface_codes.jl")

export construction_a_css_gkp, load_sparse_quantum_code, enlarge_css_generators, write_enlarged_sparse_quantum_code
include("code_constructors/sparse_quantum_codes.jl")

export LocalSearch, local_search!
include("lattice_osd/local_search.jl")

# BP tools
export initialize_tanner_graph, initialize_tanner_graph_quant, run_belief_propagation!, LDLCDecoder, run_decoder!, run_decoder_parallel!
# include("bp_algorithms/parallel_bp.jl")
include("bp_algorithms/parallel_bp_log_weight.jl")

export run_serial_belief_propagation!, run_decoder_serial!
include("bp_algorithms/serial_bp_log_weight.jl")

include("bp_algorithms/lsd_allocations.jl")
include("bp_algorithms/lsd_allocations_decoder.jl")
include("bp_algorithms/lsd_allocations_runner.jl")

include("bp_algorithms/quantized_decoder_parallel.jl")
include("bp_algorithms/ldlc_decoder.jl")

# LDLC tools
export classical_ldlc, generator_matrix, encode, decode, encode!, decode!
include("code_constructors/classical_ldlc.jl")

# Utilities
export sample_error, hard_decision, count_symbol_errors, lattice_capacity_var, lattice_capacity_std, snr, snr_db, symplectic_form, random_bitstring!, agresti_coull_confidence_interval
include("utilities/utilities.jl")

export ClassicalDecodingProblem, QuantumDecodingProblem, BinomialEstimate
export estimate_symbol_error_rate!, estimate_logical_error_rate!
include("simulation.jl")


# export l1_minimize
# include("lp_decoding/lp_decoding.jl")

end # module LatticeDecoder
