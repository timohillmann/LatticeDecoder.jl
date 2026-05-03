using Distributed
using JLD2
using LinearAlgebra
using Logging
using LatticeDecoder

include("QuantumDecodingExperiments.jl")
using .QuantumDecodingExperiments

const EXPERIMENTS_DIR = @__DIR__
const DEFAULT_QLDLC_CODES_PATH = normpath(joinpath(EXPERIMENTS_DIR, "generated_qLDLCs"))
const DEFAULT_QLDLC_RESULTS_PATH = "results/qldlc_decoding.csv"

const QLDLC_RUN_SETTINGS = (
    source_root = DEFAULT_QLDLC_CODES_PATH,
    results_path = DEFAULT_QLDLC_RESULTS_PATH,
    start_code_index = 1,
    end_code_index = nothing,
    max_codes = nothing,
    n_samples = 10,
    repeats = 1,
    target_workers = 4,
    sigmas = [0.3, 0.4],
    sigmas_already_normalized = false,
    decoder = "nearest",
    schedule = "serial",
    decoding_style = "received_vector",
    search_radius = 1.0,
    local_search = true,
    local_search_lll = false,
    sphere_decoding = false,
    full_basis = false,
)

# These generated qLDLC codes were saved by a newer JLD2 that represented BigInt fields
# inside Rational{BigInt} as base-62 strings. JLD2 0.4 can read the container,
# but it needs this conversion method to reconstruct the rationals.
function JLD2.rconvert(
    ::Type{Rational{BigInt}},
    x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}},
)
    return parse(BigInt, x.num, base = 62) // parse(BigInt, x.den, base = 62)
end

function _qldlc_value(
    x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}},
)
    return JLD2.rconvert(Rational{BigInt}, x)
end

_qldlc_value(x::AbstractMatrix) = Matrix(_qldlc_value.(x))
_qldlc_value(x::Adjoint) = Matrix(_qldlc_value(parent(x))')
_qldlc_value(x::Transpose) = transpose(_qldlc_value(parent(x)))
_qldlc_value(x::AbstractVector) = _qldlc_value.(x)
_qldlc_value(x::AbstractDict) = Dict(string(k) => _qldlc_value(v) for (k, v) in pairs(x))
_qldlc_value(x) = x

function initialize_worker_context!()
    pids = workers()
    isempty(pids) && return nothing

    module_path = joinpath(EXPERIMENTS_DIR, "QuantumDecodingExperiments.jl")
    Distributed.remotecall_eval(
        Main,
        pids,
        quote
            using LinearAlgebra
            using LatticeDecoder
            include($module_path)
            using .QuantumDecodingExperiments
        end,
    )
    return nothing
end

function ensure_worker_count(target_workers::Int)
    target_workers >= 1 || error("target_workers must be at least 1.")
    workers_to_add = max(0, target_workers - nworkers())
    workers_to_add > 0 && addprocs(workers_to_add)
    initialize_worker_context!()
    return nothing
end

function qldlc_code_files(path::AbstractString = DEFAULT_QLDLC_CODES_PATH)
    isdir(path) || error("qLDLC code directory does not exist: $(path)")
    files = filter(file -> endswith(file, ".jld2"), readdir(path; join = true))
    sort!(files)
    isempty(files) && error("No .jld2 qLDLC code files found in: $(path)")
    return files
end

function load_qldlc_code(path::AbstractString; quiet::Bool = true)
    isfile(path) || error("qLDLC code file does not exist: $(path)")

    data = if quiet
        with_logger(NullLogger()) do
            JLD2.load(path, "single_stored_object")
        end
    else
        JLD2.load(path, "single_stored_object")
    end

    data isa AbstractDict || error("Expected Dict payload in $(path), got $(typeof(data)).")
    return _qldlc_value(data)
end

function parse_qldlc_stem(stem::String)
    m = match(r"reduced_ldlc_gkp_n_(\d+)_(\d+)$", stem)
    m === nothing && return nothing
    return (
        n = parse(Int, m.captures[1]),
        code_index = parse(Int, m.captures[2]),
    )
end

function find_qldlc_generated_codes(
    source_root::String = DEFAULT_QLDLC_CODES_PATH;
    start_code_index::Int = 1,
    end_code_index::Union{Nothing,Int} = nothing,
    max_codes::Union{Nothing,Int} = nothing,
)
    generated_codes = Dict{Symbol,Any}[]

    for path in qldlc_code_files(source_root)
        stem = splitext(basename(path))[1]
        parsed = parse_qldlc_stem(stem)
        parsed === nothing && error("Could not parse qLDLC filename: $(path)")

        push!(
            generated_codes,
            Dict{Symbol,Any}(
                :jld2_path => path,
                :n => parsed.n,
                :task_id => "qldlc",
                :attempt => 1,
                :code_index => parsed.code_index,
                :code_name => stem,
                :generator_key => "qubit_generator_lll",
            ),
        )
    end

    sort!(
        generated_codes,
        by = generated_code -> (generated_code[:n], generated_code[:code_index], generated_code[:code_name]),
    )

    for (idx, generated_code) in enumerate(generated_codes)
        generated_code[:global_generated_code_index] = idx
    end

    return select_generated_codes(
        generated_codes;
        start_code_index = start_code_index,
        end_code_index = end_code_index,
        max_codes = max_codes,
    )
end

function qldlc_generated_code_path(name_or_path::AbstractString; source_root::String = DEFAULT_QLDLC_CODES_PATH)
    path = String(name_or_path)

    if isfile(path)
        return path
    end

    candidate = endswith(path, ".jld2") ? path : path * ".jld2"
    if isfile(candidate)
        return candidate
    end

    rooted_candidate = joinpath(source_root, basename(candidate))
    if isfile(rooted_candidate)
        return rooted_candidate
    end

    error(
        "Could not find generated qLDLC code for $(name_or_path). " *
        "Tried $(path), $(candidate), and $(rooted_candidate).",
    )
end

function qldlc_generated_code(name_or_path::AbstractString; source_root::String = DEFAULT_QLDLC_CODES_PATH)
    path = qldlc_generated_code_path(name_or_path; source_root = source_root)
    stem = splitext(basename(path))[1]
    parsed = parse_qldlc_stem(stem)
    parsed === nothing && error("Could not parse qLDLC filename: $(path)")

    return Dict{Symbol,Any}(
        :jld2_path => path,
        :n => parsed.n,
        :task_id => "qldlc",
        :attempt => 1,
        :code_index => parsed.code_index,
        :code_name => stem,
        :generator_key => "qubit_generator_lll",
        :global_generated_code_index => parsed.code_index,
    )
end

function select_generated_codes(
    generated_codes::Vector{Dict{Symbol,Any}};
    start_code_index::Int,
    end_code_index::Union{Nothing,Int},
    max_codes::Union{Nothing,Int},
)
    total_codes = length(generated_codes)
    total_codes == 0 && return generated_codes

    start_code_index >= 1 || error("start_code_index must be >= 1.")
    start_code_index <= total_codes || error("start_code_index=$(start_code_index) exceeds available qLDLC codes ($(total_codes)).")

    final_end_index = end_code_index === nothing ? total_codes : end_code_index
    final_end_index >= start_code_index || error("end_code_index must be >= start_code_index.")
    final_end_index <= total_codes || error("end_code_index=$(final_end_index) exceeds available qLDLC codes ($(total_codes)).")

    selected = generated_codes[start_code_index:final_end_index]

    if max_codes !== nothing
        max_codes > 0 || error("max_codes must be positive when provided.")
        selected = selected[1:min(max_codes, length(selected))]
    end

    return selected
end

"""
    symplectic_form(n::Int)

Return the symplectic form for `n` modes in the `qqpp` basis.
"""
function symplectic_form(n::Int)
    return kron(Float64[0 1; -1 0], Matrix{Float64}(I, n, n))
end

function load_qldlc_generated_code(name_or_path::AbstractString)
    code = load_qldlc_code(qldlc_generated_code_path(name_or_path))

    M = Float64.(Matrix(code["classical_generator"]))

    J = symplectic_form(size(M, 2) ÷ 2)

    H = - M * J
    inv_H = inv(H)
    G = J * inv_H

    return QECProblem(Matrix(H), Matrix(G), Matrix(inv_H), Vector{Int64}([0]))
end
