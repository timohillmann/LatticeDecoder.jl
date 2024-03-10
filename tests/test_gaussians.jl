using Test
using LatticeDecoder: gaussian, MIN_VAR


function test_gaussian_assignment()
    @testset begin
        g1 = gaussian(0.5, 0.5)
        g2 = gaussian(0.5, MIN_VAR / 2)
        g3 = gaussian(-0.5, 0.1, 0.5)
        g4 = gaussian(0.5, 0.5, 0.5, 0.5)
        @test g1.mean == 0.5 && g1.var == 0.5 && g1.weight == 1.0 && g1.period == 0.0
        @test g2.mean == 0.5 && g2.var == MIN_VAR && g2.weight == 1.0 && g2.period == 0.0
        @test g3.mean == -0.5 && g3.var == 0.1 && g3.weight == 0.5 && g3.period == 0.0
        @test g4.mean == 0.5 && g4.var == 0.5 && g4.weight == 0.5 && g4.period == 0.5
    end
end


function test_gaussian_divide()

    g1 = gaussian(0.1, 0.1, 0.1)
    g2 = gaussian(-0.2, 0.2, 0.2)
    g3 = gaussian(-0.3, 0.3, 0.3)

    g12 = prod(g1, g2)
    g23 = prod(g2, g3)
    g13 = prod(g1, g3)

    g = gaussian(0.0, 0.5, 0.5)

    @testset "divide(g1, g2)" begin
        @test isapprox(divide(g12, g1), g2)
        @test isapprox(divide(g12, g2), g1)
        @test isapprox(divide(g23, g2), g3)
        @test isapprox(divide(g23, g3), g2)
        @test isapprox(divide(g13, g1), g3)
        @test isapprox(divide(g13, g3), g1)
    end

    @testset "divide!(g_out, g1, g2)" begin
        divide!(g, g12, g1)
        @test g2 ≈ g
        divide!(g, g12, g2)
        @test g1 ≈ g
        divide!(g, g23, g2)
        @test g3 ≈ g
        divide!(g, g23, g3)
        @test g2 ≈ g
        divide!(g, g13, g3)
        @test g1 ≈ g
        divide!(g, g13, g1)
        @test g3 ≈ g
    end

    @testset "divide!(g1, g2)" begin
        divide!(g12, g2)
        @test g12 ≈ g1
        divide!(g23, g3)
        @test g23 ≈ g2
        divide!(g13, g1)
        @test g13 ≈ g3
    end
end

