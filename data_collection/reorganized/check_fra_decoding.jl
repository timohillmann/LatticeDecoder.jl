include("decode_fra_codes.jl")

using JSON3
using Printf

const DEFAULT_CHECK_RESULTS_PATH = abspath(joinpath(@__DIR__, "..", "..", DEFAULT_FRA_RESULTS_PATH))

function parse_bool_arg(value::AbstractString)
    lowered = lowercase(strip(value))
    lowered in ("true", "t", "yes", "y", "1") && return true
    lowered in ("false", "f", "no", "n", "0") && return false
    error("Expected a boolean value, got $(value).")
end

function parse_optional_int_arg(value::AbstractString)
    lowered = lowercase(strip(value))
    lowered in ("nothing", "none", "null", "") && return nothing
    return parse(Int, value)
end

function parse_float_list_arg(value::AbstractString)
    items = filter(!isempty, strip.(split(value, ",")))
    isempty(items) && error("Expected at least one sigma value.")
    return parse.(Float64, items)
end

function parse_cli_args(args)
    opts = Dict{String,Any}(
        "source-root" => DEFAULT_FRA_CODES_PATH,
        "results-path" => DEFAULT_CHECK_RESULTS_PATH,
        "start-code-index" => FRA_RUN_SETTINGS.start_code_index,
        "end-code-index" => FRA_RUN_SETTINGS.end_code_index,
        "max-codes" => FRA_RUN_SETTINGS.max_codes,
        "repeats" => FRA_RUN_SETTINGS.repeats,
        "sigmas" => Float64.(FRA_RUN_SETTINGS.sigmas),
        "sigmas-already-normalized" => FRA_RUN_SETTINGS.sigmas_already_normalized,
        "decoder" => FRA_RUN_SETTINGS.decoder,
        "schedule" => FRA_RUN_SETTINGS.schedule,
        "decoding-style" => FRA_RUN_SETTINGS.decoding_style,
        "search-radius" => FRA_RUN_SETTINGS.search_radius,
        "local-search" => FRA_RUN_SETTINGS.local_search,
        "local-search-lll" => FRA_RUN_SETTINGS.local_search_lll,
        "sphere-decoding" => FRA_RUN_SETTINGS.sphere_decoding,
        "full-basis" => FRA_RUN_SETTINGS.full_basis,
        "show-limit" => 50,
    )

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

        if key in ("start-code-index", "repeats", "show-limit")
            opts[key] = parse(Int, value)
        elseif key in ("end-code-index", "max-codes")
            opts[key] = parse_optional_int_arg(value)
        elseif key == "sigmas"
            opts[key] = parse_float_list_arg(value)
        elseif key in ("sigmas-already-normalized", "local-search", "local-search-lll", "sphere-decoding", "full-basis")
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

function print_usage()
    println("""
    Usage:
      julia --project=. data_collection/reorganized/check_fra_decoding.jl [options]

    Checks whether the FRA tasks implied by decode_fra_codes.jl were written to:
      $(DEFAULT_CHECK_RESULTS_PATH)

    Common options:
      --results-path PATH
      --source-root PATH
      --start-code-index INT
      --end-code-index INT|nothing
      --max-codes INT|nothing
      --repeats INT
      --sigmas 0.4[,0.5,...]
      --sigmas-already-normalized true|false
      --show-limit INT
    """)
end

function parse_csv_record(line::AbstractString)
    fields = String[]
    field = IOBuffer()
    in_quotes = false
    i = firstindex(line)

    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            next_i = nextind(line, i)
            if in_quotes && next_i <= lastindex(line) && line[next_i] == '"'
                write(field, '"')
                i = next_i
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(field)))
        else
            write(field, c)
        end
        i = nextind(line, i)
    end

    in_quotes && error("Malformed CSV record with unclosed quote: $(line)")
    push!(fields, String(take!(field)))
    return fields
end

function read_result_metadata(path::AbstractString)
    isfile(path) || error("Results CSV does not exist: $(path)")
    rows = Dict{String,Any}[]

    open(path, "r") do io
        header = readline(io)
        header_fields = strip.(parse_csv_record(header))
        json_col = findfirst(==("json_metadata"), header_fields)
        decoder_col = findfirst(==("decoder"), header_fields)
        json_col === nothing && error("Could not find json_metadata column in $(path).")
        decoder_col === nothing && error("Could not find decoder column in $(path).")

        for (line_number, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            fields = parse_csv_record(line)
            length(fields) >= json_col || error("Line $(line_number + 1) has too few CSV fields.")
            parsed_meta = JSON3.read(fields[json_col])
            meta = Dict{String,Any}(string(k) => v for (k, v) in pairs(parsed_meta))
            meta["csv_decoder"] = strip(fields[decoder_col])
            meta["csv_line"] = line_number + 1
            push!(rows, meta)
        end
    end

    return rows
end

function expected_tasks(opts)
    artifacts = discover_fra_artifacts(
        String(opts["source-root"]);
        start_code_index = Int(opts["start-code-index"]),
        end_code_index = opts["end-code-index"],
        max_codes = opts["max-codes"],
    )
    sigmas = normalize_sigmas(
        opts["sigmas"];
        already_normalized = Bool(opts["sigmas-already-normalized"]),
    )

    tasks = Dict{String,Any}[]
    for repeat in 1:Int(opts["repeats"])
        for artifact in artifacts
            for sigma in sigmas
                push!(
                    tasks,
                    Dict{String,Any}(
                        "code_name" => artifact[:code_name],
                        "n" => artifact[:n],
                        "code_index" => artifact[:code_index],
                        "global_artifact_index" => artifact[:global_artifact_index],
                        "repeat" => repeat,
                        "sigma" => sigma,
                        "decoder" => String(opts["decoder"]),
                        "schedule" => String(opts["schedule"]),
                        "decoding_style" => String(opts["decoding-style"]),
                        "search_radius" => Float64(opts["search-radius"]),
                        "local_search" => Bool(opts["local-search"]),
                        "local_search_lll" => Bool(opts["local-search-lll"]),
                        "sphere_decoding" => Bool(opts["sphere-decoding"]),
                        "full_basis" => Bool(opts["full-basis"]),
                    ),
                )
            end
        end
    end

    return artifacts, sigmas, tasks
end

canonical_float(x) = @sprintf("%.17g", Float64(x))

function task_key(meta)
    fields = (
        "code_name",
        "n",
        "code_index",
        "global_artifact_index",
        "repeat",
        "sigma",
        "decoder",
        "schedule",
        "decoding_style",
        "search_radius",
        "local_search",
        "local_search_lll",
        "sphere_decoding",
        "full_basis",
    )

    values = String[]
    for field in fields
        value = meta[field]
        if field in ("sigma", "search_radius")
            push!(values, canonical_float(value))
        else
            push!(values, string(value))
        end
    end
    return join(values, "|")
end

function code_key(meta)
    return string(meta["global_artifact_index"], "|", meta["code_name"])
end

function main(args = ARGS)
    opts = parse_cli_args(args)
    if get(opts, "help", false)
        print_usage()
        return 0
    end

    results_path = abspath(String(opts["results-path"]))
    artifacts, sigmas, tasks = expected_tasks(opts)
    rows = read_result_metadata(results_path)

    expected_by_task = Dict(task_key(task) => task for task in tasks)
    seen_by_task = Dict{String,Vector{Dict{String,Any}}}()
    for row in rows
        get(row, "task_id", nothing) == "fra" || continue
        key = try
            task_key(row)
        catch
            continue
        end
        push!(get!(seen_by_task, key, Dict{String,Any}[]), row)
    end

    expected_keys = Set(keys(expected_by_task))
    seen_keys = Set(keys(seen_by_task))
    present_keys = intersect(expected_keys, seen_keys)
    extra_keys = setdiff(seen_keys, expected_keys)
    missing_keys = collect(setdiff(expected_keys, seen_keys))
    sort!(missing_keys, by = key -> (
        expected_by_task[key]["global_artifact_index"],
        expected_by_task[key]["repeat"],
        expected_by_task[key]["sigma"],
    ))
    duplicate_keys = [key for key in present_keys if length(seen_by_task[key]) > 1]
    sort!(duplicate_keys, by = key -> (
        expected_by_task[key]["global_artifact_index"],
        expected_by_task[key]["repeat"],
        expected_by_task[key]["sigma"],
    ))
    present_row_count = sum(length(seen_by_task[key]) for key in present_keys; init = 0)

    expected_code_keys = Set(code_key(Dict{String,Any}(
        "global_artifact_index" => artifact[:global_artifact_index],
        "code_name" => artifact[:code_name],
    )) for artifact in artifacts)
    seen_code_keys = Set{String}()
    for row in rows
        get(row, "task_id", nothing) == "fra" || continue
        haskey(row, "global_artifact_index") || continue
        haskey(row, "code_name") || continue
        push!(seen_code_keys, code_key(row))
    end
    missing_code_keys = collect(setdiff(expected_code_keys, seen_code_keys))
    sort!(missing_code_keys, by = key -> parse(Int, split(key, "|"; limit = 2)[1]))

    println("FRA decoding check")
    println("  results_path: $(results_path)")
    println("  source_root:  $(opts["source-root"])")
    println("  artifacts:    $(length(artifacts))")
    println("  repeats:      $(opts["repeats"])")
    println("  sigmas:       $(join(canonical_float.(sigmas), ", "))")
    println("  expected task rows:       $(length(tasks))")
    println("  expected rows present:    $(length(present_keys))")
    println("  matching CSV rows:        $(present_row_count)")
    println("  missing task rows:        $(length(missing_keys))")
    println("  duplicate expected tasks: $(length(duplicate_keys))")
    println("  rows for other settings:  $(sum(length(seen_by_task[key]) for key in extra_keys; init = 0))")
    println("  missing codes:            $(length(missing_code_keys))")

    show_limit = Int(opts["show-limit"])
    if !isempty(missing_keys)
        println()
        println("Missing task rows:")
        for key in Iterators.take(missing_keys, show_limit)
            task = expected_by_task[key]
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
        length(missing_keys) > show_limit && println("  ... $(length(missing_keys) - show_limit) more")
    end

    if !isempty(missing_code_keys)
        by_code_key = Dict(
            code_key(Dict{String,Any}(
                "global_artifact_index" => artifact[:global_artifact_index],
                "code_name" => artifact[:code_name],
            )) => artifact for artifact in artifacts
        )
        println()
        println("Codes with no FRA rows at all:")
        for key in Iterators.take(missing_code_keys, show_limit)
            artifact = by_code_key[key]
            println("  ", artifact[:global_artifact_index], ": ", artifact[:code_name])
        end
        length(missing_code_keys) > show_limit && println("  ... $(length(missing_code_keys) - show_limit) more")
    end

    if !isempty(duplicate_keys)
        println()
        println("Duplicate matching task rows:")
        for key in Iterators.take(duplicate_keys, show_limit)
            task = expected_by_task[key]
            lines = [row["csv_line"] for row in seen_by_task[key]]
            println(
                "  ",
                task["global_artifact_index"],
                ": ",
                task["code_name"],
                " repeat=",
                task["repeat"],
                " sigma=",
                canonical_float(task["sigma"]),
                " lines=",
                join(lines, ","),
            )
        end
        length(duplicate_keys) > show_limit && println("  ... $(length(duplicate_keys) - show_limit) more")
    end

    if isempty(missing_keys) && isempty(duplicate_keys)
        println()
        println("All expected FRA decoding task rows are present exactly once.")
        return 0
    end

    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
