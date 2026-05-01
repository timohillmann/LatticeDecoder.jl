include("check_fra_decoding.jl")

const MISSING_FRA_RUN_SETTINGS = (
    source_root = DEFAULT_FRA_CODES_PATH,
    results_path = DEFAULT_CHECK_RESULTS_PATH,
    start_code_index = FRA_RUN_SETTINGS.start_code_index,
    end_code_index = FRA_RUN_SETTINGS.end_code_index,
    max_codes = FRA_RUN_SETTINGS.max_codes,
    n_samples = FRA_RUN_SETTINGS.n_samples,
    repeats = FRA_RUN_SETTINGS.repeats,
    target_workers = FRA_RUN_SETTINGS.target_workers,
    sigmas = Float64.(FRA_RUN_SETTINGS.sigmas),
    sigmas_already_normalized = FRA_RUN_SETTINGS.sigmas_already_normalized,
    decoder = FRA_RUN_SETTINGS.decoder,
    schedule = FRA_RUN_SETTINGS.schedule,
    decoding_style = FRA_RUN_SETTINGS.decoding_style,
    search_radius = FRA_RUN_SETTINGS.search_radius,
    local_search = FRA_RUN_SETTINGS.local_search,
    local_search_lll = FRA_RUN_SETTINGS.local_search_lll,
    sphere_decoding = FRA_RUN_SETTINGS.sphere_decoding,
    full_basis = FRA_RUN_SETTINGS.full_basis,
    show_limit = 50,
    dry_run = false,
)

function missing_fra_settings_dict(settings = MISSING_FRA_RUN_SETTINGS)
    return Dict{String,Any}(
        "source-root" => settings.source_root,
        "results-path" => settings.results_path,
        "start-code-index" => settings.start_code_index,
        "end-code-index" => settings.end_code_index,
        "max-codes" => settings.max_codes,
        "n-samples" => settings.n_samples,
        "repeats" => settings.repeats,
        "target-workers" => settings.target_workers,
        "sigmas" => Float64.(settings.sigmas),
        "sigmas-already-normalized" => settings.sigmas_already_normalized,
        "decoder" => settings.decoder,
        "schedule" => settings.schedule,
        "decoding-style" => settings.decoding_style,
        "search-radius" => settings.search_radius,
        "local-search" => settings.local_search,
        "local-search-lll" => settings.local_search_lll,
        "sphere-decoding" => settings.sphere_decoding,
        "full-basis" => settings.full_basis,
        "show-limit" => settings.show_limit,
        "dry-run" => settings.dry_run,
    )
end

function parse_missing_decode_args(args)
    opts = missing_fra_settings_dict()

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            opts["help"] = true
            i += 1
            continue
        end

        startswith(arg, "--") || error("Unexpected argument $(arg). Use --help for usage.")
        key_value = split(arg[3:end], "="; limit = 2)
        key = key_value[1]
        haskey(opts, key) || error("Unknown option --$(key). Use --help for usage.")

        if length(key_value) == 2
            value = key_value[2]
        else
            i < length(args) || error("Missing value for --$(key).")
            i += 1
            value = args[i]
        end

        if key in ("start-code-index", "n-samples", "repeats", "target-workers", "show-limit")
            opts[key] = parse(Int, value)
        elseif key in ("end-code-index", "max-codes")
            opts[key] = parse_optional_int_arg(value)
        elseif key == "sigmas"
            opts[key] = parse_float_list_arg(value)
        elseif key in ("sigmas-already-normalized", "local-search", "local-search-lll", "sphere-decoding", "full-basis", "dry-run")
            opts[key] = parse_bool_arg(value)
        elseif key == "search-radius"
            opts[key] = parse(Float64, value)
        else
            opts[key] = value
        end

        i += 1
    end

    return opts
end

function print_missing_decode_usage()
    println("""
    Usage:
      julia --project=. data_collection/reorganized/decode_missing_fra_codes.jl [options]

    Decodes exactly the FRA task rows missing from:
      $(DEFAULT_CHECK_RESULTS_PATH)

    Common options:
      --results-path PATH
      --source-root PATH
      --n-samples INT
      --target-workers INT
      --sigmas 0.4[,0.5,...]
      --sigmas-already-normalized true|false
      --dry-run true|false
      --show-limit INT
    """)
end

function find_missing_fra_tasks(opts)
    results_path = abspath(String(opts["results-path"]))
    artifacts, sigmas, tasks = expected_tasks(opts)
    rows = isfile(results_path) ? read_result_metadata(results_path) : Dict{String,Any}[]

    expected_by_task = Dict(task_key(task) => task for task in tasks)
    seen_keys = Set{String}()
    for row in rows
        get(row, "task_id", nothing) == "fra" || continue
        key = try
            task_key(row)
        catch
            continue
        end
        push!(seen_keys, key)
    end

    missing_keys = collect(setdiff(Set(keys(expected_by_task)), seen_keys))
    sort!(missing_keys, by = key -> (
        expected_by_task[key]["global_artifact_index"],
        expected_by_task[key]["repeat"],
        expected_by_task[key]["sigma"],
    ))

    return artifacts, sigmas, [expected_by_task[key] for key in missing_keys]
end

function decode_missing_fra_task!(
    task::Dict{String,Any},
    artifact::Dict{Symbol,Any},
    opts::Dict{String,Any},
)
    sigma = Float64(task["sigma"])
    p = Dict{Symbol,Any}(
        :search_radius => Float64(opts["search-radius"]),
        :decoder => String(opts["decoder"]),
        :local_search => Bool(opts["local-search"]),
        :local_search_lll => Bool(opts["local-search-lll"]),
        :schedule => String(opts["schedule"]),
        :decoding_style => String(opts["decoding-style"]),
        :sigmas => [sigma],
        :sphere_decoding => Bool(opts["sphere-decoding"]),
        :full_basis => Bool(opts["full-basis"]),
        :n => artifact[:n],
        :code_name => artifact[:code_name],
        :task_id => artifact[:task_id],
        :attempt => artifact[:attempt],
        :code_index => artifact[:code_index],
        :global_artifact_index => artifact[:global_artifact_index],
    )

    results, H, inv_residual = run_fra_code_experiment(p, artifact, Int64(opts["n-samples"]))

    for (σ, res) in zip(p[:sigmas], results)
        meta = build_metadata(
            p,
            σ,
            H;
            repeat = Int(task["repeat"]),
            code_file = artifact[:jld2_path],
            source_type = "fra_jld2",
            generator_key = artifact[:generator_key],
            matrix_inverse_residual = inv_residual,
        )

        add_data!(
            String(opts["results-path"]);
            shots = Int(opts["n-samples"]),
            errors = res,
            decoder = string(p[:decoder]),
            json_metadata = meta,
        )
    end
end

function run_missing_fra_codes_decoding(opts::Dict{String,Any})
    opts["results-path"] = abspath(String(opts["results-path"]))
    results_path = String(opts["results-path"])
    artifacts, sigmas, missing_tasks = find_missing_fra_tasks(opts)
    artifacts_by_index = Dict(artifact[:global_artifact_index] => artifact for artifact in artifacts)

    println("Missing FRA decode")
    println("  results_path:   $(results_path)")
    println("  source_root:    $(opts["source-root"])")
    println("  artifacts:      $(length(artifacts))")
    println("  n_samples:      $(opts["n-samples"])")
    println("  target_workers: $(opts["target-workers"])")
    println("  sigmas:         $(join(canonical_float.(sigmas), ", "))")
    println("  missing tasks:  $(length(missing_tasks))")

    show_limit = Int(opts["show-limit"])
    for task in Iterators.take(missing_tasks, show_limit)
        println(
            "  ",
            task["global_artifact_index"],
            ": ",
            task["code_name"],
            " repeat=",
            task["repeat"],
            " sigma=",
            canonical_float(task["sigma"]),
        )
    end
    length(missing_tasks) > show_limit && println("  ... $(length(missing_tasks) - show_limit) more")

    if isempty(missing_tasks)
        println()
        println("No missing FRA task rows found.")
        return 0
    end

    if Bool(opts["dry-run"])
        println()
        println("Dry run only; no decoding performed.")
        return 0
    end

    ensure_worker_count(Int(opts["target-workers"]))

    for (idx, task) in enumerate(missing_tasks)
        artifact = artifacts_by_index[task["global_artifact_index"]]
        println()
        println(
            "Decoding missing task $(idx) / $(length(missing_tasks)): ",
            artifact[:code_name],
            " repeat=",
            task["repeat"],
            " sigma=",
            canonical_float(task["sigma"]),
        )

        try
            decode_missing_fra_task!(task, artifact, opts)
        catch err
            @warn "Skipping failed missing FRA task." code_file = artifact[:jld2_path] repeat = task["repeat"] sigma = task["sigma"] exception = (err, catch_backtrace())
        end

        flush(stdout)
    end

    println()
    println("Rechecking missing FRA tasks...")
    _, _, remaining_tasks = find_missing_fra_tasks(opts)
    println("  remaining missing tasks: $(length(remaining_tasks))")
    return isempty(remaining_tasks) ? 0 : 1
end

function run_missing_fra_with_script_settings(settings = MISSING_FRA_RUN_SETTINGS)
    return run_missing_fra_codes_decoding(missing_fra_settings_dict(settings))
end

function main(args = ARGS)
    opts = parse_missing_decode_args(args)
    if get(opts, "help", false)
        print_missing_decode_usage()
        return 0
    end

    return run_missing_fra_codes_decoding(opts)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_missing_fra_with_script_settings())
end
