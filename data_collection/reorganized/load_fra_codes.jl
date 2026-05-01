using JLD2
using LinearAlgebra
using Logging

const DEFAULT_FRA_CODES_PATH = normpath(joinpath(@__DIR__, "..", "..", "data", "generator_matrices", "ldlc", "reduced_ldlc"))

# These FRA artifacts were saved by a newer JLD2 that represented BigInt fields
# inside Rational{BigInt} as base-62 strings. JLD2 0.4 can read the container,
# but it needs this conversion method to reconstruct the rationals.
function JLD2.rconvert(
    ::Type{Rational{BigInt}},
    x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}},
)
    return parse(BigInt, x.num, base = 62) // parse(BigInt, x.den, base = 62)
end

function _fra_value(
    x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}},
)
    return JLD2.rconvert(Rational{BigInt}, x)
end

_fra_value(x::AbstractMatrix) = Matrix(_fra_value.(x))
_fra_value(x::Adjoint) = Matrix(_fra_value(parent(x))')
_fra_value(x::Transpose) = transpose(_fra_value(parent(x)))
_fra_value(x::AbstractVector) = _fra_value.(x)
_fra_value(x::AbstractDict) = Dict(string(k) => _fra_value(v) for (k, v) in pairs(x))
_fra_value(x) = x

function fra_code_files(path::AbstractString = DEFAULT_FRA_CODES_PATH)
    isdir(path) || error("FRA code directory does not exist: $(path)")
    files = filter(p -> endswith(p, ".jld2"), readdir(path; join = true))
    sort!(files)
    isempty(files) && error("No .jld2 FRA code files found in: $(path)")
    return files
end

function load_fra_code(path::AbstractString; quiet::Bool = true)
    isfile(path) || error("FRA code file does not exist: $(path)")
    data = if quiet
        with_logger(NullLogger()) do
            JLD2.load(path, "single_stored_object")
        end
    else
        JLD2.load(path, "single_stored_object")
    end
    data isa AbstractDict || error("Expected Dict payload in $(path), got $(typeof(data)).")
    return _fra_value(data)
end

function load_fra_codes(
    path::AbstractString = DEFAULT_FRA_CODES_PATH;
    max_codes::Union{Nothing,Int} = nothing,
    quiet::Bool = true,
)
    files = fra_code_files(path)
    if max_codes !== nothing
        max_codes > 0 || error("max_codes must be positive when provided.")
        files = files[1:min(max_codes, length(files))]
    end
    return [load_fra_code(file; quiet = quiet) for file in files]
end

if abspath(PROGRAM_FILE) == @__FILE__
    codes = load_fra_codes(; max_codes = 1)
    first_code = first(codes)
    println("Loaded $(length(codes)) FRA code(s) from $(DEFAULT_FRA_CODES_PATH)")
    println("First code keys: $(collect(keys(first_code)))")
end
