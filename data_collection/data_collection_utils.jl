if !isdefined(@__MODULE__, :DEFAULT_MAX_ERRORS)
    const DEFAULT_MAX_ERRORS = 500
end

function _positive_int_env(name::AbstractString)
    haskey(ENV, name) || return nothing

    value = tryparse(Int, ENV[name])
    value === nothing && return nothing
    value > 0 || return nothing
    return value
end

function target_worker_count(default::Int = 1)
    for name in ("JULIA_NUM_WORKERS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
        value = _positive_int_env(name)
        value === nothing || return value
    end

    return max(default, Sys.CPU_THREADS)
end

function active_worker_process_count()
    return nprocs() == 1 ? 0 : nworkers()
end

function ensure_local_worker_count!(target::Int)
    target = max(1, target)
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

    return nworkers()
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
