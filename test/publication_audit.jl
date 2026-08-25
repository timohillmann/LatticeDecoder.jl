using LinearAlgebra
using NPZ
using SHA
using Test

const PUBLIC_ROOT = normpath(joinpath(@__DIR__, ".."))

function _binary_bivariate_circulant(ell::Int, m::Int, terms)
    n = ell * m
    result = zeros(Int, n, n)
    for row in 0:(n - 1)
        x_index, y_index = divrem(row, m)
        for (x_power, y_power) in terms
            column = mod(x_index + x_power, ell) * m + mod(y_index + y_power, m)
            result[row + 1, column + 1] = mod(result[row + 1, column + 1] + 1, 2)
        end
    end
    return result
end

function test_fixture_construction_provenance()
    qldlc = npzread(joinpath(PUBLIC_ROOT, "examples", "fixtures", "qldlc_n13_3.npz"))
    modes = 13
    identity = Matrix{Float64}(I, modes, modes)
    zero_block = zeros(modes, modes)
    symplectic = [zero_block identity; -identity zero_block]
    classical = qldlc["classical_generator"]
    reduced = qldlc["reduced_generator"]
    logical_rows = qldlc["logical_rows"]
    decision = -reduced * symplectic
    @test qldlc["decision_H"] ≈ decision
    @test qldlc["G"] ≈ inv(decision)
    @test qldlc["logical_check"] ≈ inv(reduced)
    @test qldlc["bp_H"] ≈ vcat(-classical * symplectic, -2logical_rows * symplectic)

    instances = (
        (file="30_4_5_p2.npz", ell=5, m=3, b_terms=((0, 0), (0, 1), (2, 2))),
        (file="48_4_7_p2.npz", ell=8, m=3, b_terms=((0, 0), (0, 1), (3, 2))),
    )
    for instance in instances
        data = npzread(joinpath(
            PUBLIC_ROOT,
            "examples",
            "fixtures",
            "bivariate_bicycle",
            instance.file,
        ))
        A = _binary_bivariate_circulant(instance.ell, instance.m, ((0, 0), (1, 0)))
        B = _binary_bivariate_circulant(instance.ell, instance.m, instance.b_terms)
        @test data["hx"] == hcat(A, B)
        @test data["hz"] == hcat(transpose(B), transpose(A))
    end
end

function public_paths()
    command = Cmd(`git ls-files`; dir=PUBLIC_ROOT)
    paths = filter(!isempty, split(read(command, String), '\n'))
    if isempty(paths)
        # Supports auditing an orphan worktree before its first local commit.
        for (directory, subdirectories, files) in walkdir(PUBLIC_ROOT)
            filter!(name -> name != ".git", subdirectories)
            for file in files
                path = relpath(joinpath(directory, file), PUBLIC_ROOT)
                path == ".git" || push!(paths, path)
            end
        end
        sort!(paths)
    end
    return paths
end

function test_publication_audit()
    test_fixture_construction_provenance()
    paths = public_paths()
    forbidden_directories = (
        "data_collection/",
        "data_plotting/",
        "experiments/",
        "research/",
        "slurm/",
        "results/",
        "logs/",
        "generator_matrices/",
        ".vscode/",
    )
    forbidden_suffixes = (".csv", ".tsv", ".out", ".sbatch")

    @test !isempty(paths)
    @test all(path -> !any(prefix -> startswith(path, prefix), forbidden_directories), paths)
    @test all(path -> !any(suffix -> endswith(lowercase(path), suffix), forbidden_suffixes), paths)
    @test all(path -> basename(path) != ".DS_Store", paths)

    notebooks = sort(filter(path -> startswith(path, "examples/") && endswith(path, ".ipynb"), paths))
    @test length(notebooks) == 6
    @test notebooks == sort([
        "examples/bivariate_bicycle_css_decode.ipynb",
        "examples/classical_ldlc_decode.ipynb",
        "examples/qldlc_generated_code_decode.ipynb",
        "examples/rep_code_css_decode.ipynb",
        "examples/surface_code_css_decode.ipynb",
        "examples/surface_code_noncss_decode.ipynb",
    ])

    home_fragment = join(["", "Users", ""] , '/')
    stale_example_fragment = join(["examples", "new"], '_')
    text_suffixes = (".jl", ".toml", ".md", ".yml", ".yaml", ".cff", ".json", ".ipynb")
    for path in filter(path -> any(suffix -> endswith(path, suffix), text_suffixes), paths)
        contents = read(joinpath(PUBLIC_ROOT, path), String)
        @test !occursin(home_fragment, contents)
        @test !occursin(stale_example_fragment, contents)
        @test !occursin(r"AKIA[0-9A-Z]{16}", contents)
        @test !occursin("ghp" * "_", contents)
        @test !occursin("BEGIN " * "PRIVATE KEY", contents)
    end

    for notebook in notebooks
        path = joinpath(PUBLIC_ROOT, notebook)
        contents = read(path, String)
        @test filesize(path) < 200_000
        @test !occursin("\"output_type\": \"error\"", contents)
        @test !occursin("image/png", contents)
    end

    checksums = Dict(
        "examples/fixtures/qldlc_n13_3.npz" =>
            "0320e486a1b81c77453ce0c908e289ad930034b388955d5271d421589205de42",
        "examples/fixtures/bivariate_bicycle/30_4_5_p2.npz" =>
            "72bc37894559a83efac2cf618f83dc469ba004a427b885e76e13d0fe6817e0bc",
        "examples/fixtures/bivariate_bicycle/48_4_7_p2.npz" =>
            "e5d2a91bf5287afc9b7189f9975c8ce6b5957d6ab99bf662c35b6c8e5c821614",
        "test/fixtures/ldlc_n128_d5_H.npy" =>
            "e356c7573c0f6adf80bd27ed1a56fe8f2c70e8ba7eb2f7ed356e5ecc49dbb161",
        "test/fixtures/ldlc_n128_d5_decoder_goldens.npz" =>
            "101e6b4c3fa55b82c544c84f80ad6d8e4d0b3593117e412d17bae1ae18cdf1bf",
    )
    for (path, expected) in checksums
        full_path = joinpath(PUBLIC_ROOT, path)
        @test path in paths
        @test bytes2hex(sha256(read(full_path))) == expected
    end

    @test "examples/fixtures/README.md" in paths
    @test "test/fixtures/README.md" in paths

    upstream_revision = "152d72e1e0195ab14dde0679abf2a5b6b8288a1d"
    upstream_url = "https://github.com/amazon-science/LatticeAlgorithms.jl.git"
    for manifest in ("Manifest.toml", "Manifest-v1.12.toml")
        @test manifest in paths
        contents = read(joinpath(PUBLIC_ROOT, manifest), String)
        @test occursin("repo-rev = \"$upstream_revision\"", contents)
        @test occursin("repo-url = \"$upstream_url\"", contents)
    end
end
