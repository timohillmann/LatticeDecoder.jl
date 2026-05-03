module QuantumDecodingExperiments

using Distributed
using LinearAlgebra
using LatticeDecoder

export QECProblem,
    build_metadata,
    is_not_logical_error,
    parameter_grid,
    qec_experiment,
    qec_sample,
    run_experiment_grid,
    run_surface_code_css_experiment,
    run_surface_code_experiment

include("shared.jl")
include("codes/surface_code.jl")

end
