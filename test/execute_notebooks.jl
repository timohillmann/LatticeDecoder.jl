using JSON3
using Test

const NOTEBOOK_ROOT = normpath(joinpath(@__DIR__, ".."))
const NOTEBOOKS = [
    "bivariate_bicycle_css_decode.ipynb",
    "classical_ldlc_decode.ipynb",
    "qldlc_generated_code_decode.ipynb",
    "rep_code_css_decode.ipynb",
    "surface_code_css_decode.ipynb",
    "surface_code_noncss_decode.ipynb",
]

function notebook_source(path::AbstractString)
    document = JSON3.read(read(path, String))
    cells = String[]
    for cell in document.cells
        cell.cell_type == "code" || continue
        push!(cells, join(String.(cell.source)))
    end
    return join(cells, "\n\n")
end

function test_notebook_smoke()
    mktempdir() do temporary_directory
        for notebook in NOTEBOOKS
            notebook_path = joinpath(NOTEBOOK_ROOT, "examples", notebook)
            script_path = joinpath(temporary_directory, replace(notebook, ".ipynb" => ".jl"))
            write(script_path, notebook_source(notebook_path))
            command = Cmd(
                `$(Base.julia_cmd()) --startup-file=no --project=$(NOTEBOOK_ROOT) $(script_path)`;
                dir=NOTEBOOK_ROOT,
            )
            @test begin
                try
                    run(addenv(
                        command,
                        "LATTICEDECODER_EXAMPLE_SAMPLES" => "1",
                        "JULIA_LOAD_PATH" => "@:@stdlib",
                    ))
                    true
                catch error
                    showerror(stderr, error)
                    println(stderr, "\nNotebook failed: ", notebook)
                    false
                end
            end
        end
    end
end
