if !isdefined(@__MODULE__, :DEFAULT_MAX_ERRORS)
    const DEFAULT_MAX_ERRORS = 500
end

function _nonnegative_int_env(name::AbstractString)
    haskey(ENV, name) || return nothing

    value = tryparse(Int, ENV[name])
    value === nothing && return nothing
    value >= 0 || return nothing
    return value
end

function _positive_int_env(name::AbstractString, default::Int)
    value = _nonnegative_int_env(name)
    value === nothing && return default
    value > 0 || return default
    return value
end

function _float_list_env(name::AbstractString, default::Vector{Float64})
    haskey(ENV, name) || return default

    values = Float64[]
    for raw in split(ENV[name], ",")
        value = tryparse(Float64, strip(raw))
        value === nothing && return default
        push!(values, value)
    end

    return isempty(values) ? default : values
end

function _string_list_env(name::AbstractString, default::Vector{String})
    haskey(ENV, name) || return default

    values = [strip(raw) for raw in split(ENV[name], ",") if !isempty(strip(raw))]
    return isempty(values) ? default : values
end

function _bool_env(name::AbstractString, default::Bool = false)
    haskey(ENV, name) || return default
    value = lowercase(strip(ENV[name]))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    return default
end

sweep_samples(default::Int = 5_000) = _positive_int_env("LSD_SWEEP_SAMPLES", default)
sweep_repeats(default::Int = 250) = _positive_int_env("LSD_SWEEP_REPEATS", default)
function sweep_max_errors(default::Int = 200)
    value = _nonnegative_int_env("LSD_SWEEP_MAX_ERRORS")
    return value === nothing ? default : value
end
sweep_betas(default::Vector{Float64} = collect(1.5:0.1:6.0)) = _float_list_env("LSD_SWEEP_BETAS", default)
sweep_w_mins(default::Vector{Float64} = collect(0.5:0.05:1.25)) = _float_list_env("LSD_SWEEP_W_MINS", default)
sweep_code_names(default::Vector{String}) = _string_list_env("LSD_SWEEP_CODE_NAMES", default)
sweep_output_path(default::AbstractString) = get(ENV, "LSD_SWEEP_OUTPUT_PATH", default)
sweep_dry_run(default::Bool = false) = _bool_env("LSD_SWEEP_DRY_RUN", default)

function target_worker_count(default::Int = 1)
    for name in ("JULIA_NUM_WORKERS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
        value = _nonnegative_int_env(name)
        value === nothing || return value
    end

    return max(default, Sys.CPU_THREADS)
end

function data_collection_gc_interval(default::Int = 100)
    value = _nonnegative_int_env("DATA_COLLECTION_GC_INTERVAL")
    value === nothing && return default
    return value
end

function active_worker_process_count()
    return nprocs() == 1 ? 0 : nworkers()
end

function ensure_local_worker_count!(target::Int)
    target = max(0, target)
    current = active_worker_process_count()

    if current < target
        addprocs(target - current)
    end

    return active_worker_process_count()
end

function ensure_data_collection_workers!(target::Int = target_worker_count())
    ensure_local_worker_count!(target)

    @sync for pid in workers()
        @async remotecall_wait(include, pid, @__FILE__)
    end

    return active_worker_process_count()
end

function data_collection_sample_sum(f::Function, n_samples::Int)
    gc_interval = data_collection_gc_interval()

    if active_worker_process_count() == 0
        total = 0
        for sample_index in 1:n_samples
            total += f(sample_index)
            if gc_interval > 0 && sample_index % gc_interval == 0
                GC.gc()
            end
        end
        return total
    end

    return @distributed (+) for sample_index in 1:n_samples
        result = f(sample_index)
        if gc_interval > 0 && sample_index % gc_interval == 0
            GC.gc()
        end
        result
    end
end

function is_not_logical_error(
    logical_check::AbstractMatrix{Float64},
    residual::Vector{Float64};
    atol::Float64 = 1e-5,
)
    logical_coordinates = logical_check' * residual
    return all(abs(x - round(x)) < atol for x in logical_coordinates)
end

function parameter_grid(param_ranges::Dict)
    keys_list = collect(keys(param_ranges))
    values_list = collect(values(param_ranges))

    return [
        Dict(zip(keys_list, combo))
        for combo in Iterators.product(values_list...)
        if is_active_parameter_combination(Dict(zip(keys_list, combo)))
    ]
end

function is_active_parameter_combination(params::Dict)
    if haskey(params, :local_search) &&
       haskey(params, :sphere_decoding) &&
       !params[:local_search] &&
       params[:sphere_decoding]
        return false
    end

    return true
end

function accumulated_errors_by_strong_id(path::AbstractString)
    totals = Dict{String,Int}()
    isfile(path) || return totals

    for (line_number, line) in enumerate(eachline(path))
        line_number == 1 && continue
        isempty(strip(line)) && continue

        fields = split(line, ","; limit = 7)
        length(fields) >= 6 || continue

        errors = tryparse(Int, strip(fields[2]))
        errors === nothing && continue

        strong_id = strip(fields[6])
        totals[strong_id] = get(totals, strong_id, 0) + errors
    end

    return totals
end
