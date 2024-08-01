using JSON3
using SHA

const HEADER = "     shots,    errors,  discards, seconds,decoder,strong_id,json_metadata,custom_counts"


"""
    write_csv_line(file::IO,
    shots::Int,
    errors::Int,
    discards::Int,
    seconds::Float64,
    decoder::String,
    strong_id::String,
    json_metadata::Dict,
    custom_counts::Dict)

Write a line to a CSV file.
"""
function write_csv_line(file::IO,
    shots::Int,
    errors::Int,
    discards::Int,
    seconds::Float64,
    decoder::String,
    strong_id::String,
    json_metadata::Dict,
    custom_counts::Dict)

    shots = lpad(string(shots), 10)
    errors = lpad(string(errors), 10)
    discards = lpad(string(discards), 10)
    seconds = lpad(string(seconds), 8)


    json_metadata_str = escape_double_quotes(JSON3.write(json_metadata))
    custom_counts_str = escape_double_quotes(JSON3.write(custom_counts))


    println(file, """$shots,$errors,$discards,$seconds,$decoder,$strong_id,"$json_metadata_str","$custom_counts_str\"""")
end

function escape_double_quotes(s::String)
    return replace(s, "\"" => "\"\"")
end


"""
    get_strong_id_from_json(json_metadata::Dict)

Get a strong id from a json metadata dictionary.
"""
function get_strong_id_from_json(json_metadata::Dict)
    out = JSON3.write(json_metadata; canonical=true)
    return bytes2hex(sha256(out))
end



"""
    metadata(; kwargs...)

Create a metadata dictionary for the simulation data.
"""
function metadata(; kwargs...)
    return Dict(kwargs...)
end


"""
    add_data!(path::String,
    shots::Int,
    errors::Int,
    json_metadata::Dict,
    decoder::String,
    discards::Int=0,
    seconds::Float64=0.0,
    custom_counts::Dict=Dict())

Add simulation data to file. If the file does not exist, create it and write the header.
"""
function add_data!(path::String;
    shots::Int,
    errors::Int,
    json_metadata::Dict,
    decoder::String,
    discards::Int=0,
    seconds::Float64=0.0,
    custom_counts::Dict=Dict())

    # if file / folder does not exist create it and write header
    # check if folder exists
    folder = dirname(path)
    if !isdir(folder)
        mkdir(folder)
    end

    if !isfile(path)
        open(path, "w") do file
            println(file, HEADER)
        end
    end

    strong_id = get_strong_id_from_json(json_metadata)

    open(path, "a") do file
        write_csv_line(file, shots, errors, discards, seconds, decoder, strong_id, json_metadata, custom_counts)
    end
end


function test_write_csv_line()
    path = "results.csv"

    # if file does not exist create it and write header
    if !isfile(path)
        open(path, "w") do file
            println(file, HEADER)
        end
    end

    open(path, "a") do file
        write_csv_line(file, 100, 0, 0, 0.1, "lsd", strong_id, data, Dict("e" => 0, "d" => 0))
    end
end
