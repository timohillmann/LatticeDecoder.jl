using Distributed
using LaTeXStrings
if nprocs()<6
    addprocs(6-nprocs())
end
# @everywhere using LatticeDecoder
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra
using Plots
# using LatticeAlgorithms

# if true
#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/lattice_tools/overcomplete_syndrome.jl")

#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/code_constructors/rep_codes.jl")

#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/utilities/utilities.jl")

#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians.jl")
#     # include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians_log_weights_OLD.jl")
#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl")
#     @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl")
# end

# include("../src/code_constructors/rep_codes.jl")


@everywhere function sample_and_decode(code::QuantumCode,H::AbstractMatrix,J::AbstractMatrix,G::AbstractMatrix, decision_H::AbstractMatrix,logical::AbstractVector,σ::Float64,n::Int64, Mh, Vt, bottomSys)
    tg = initialize_tanner_graph(code);
    
    # println("new error")
    y= sample_error(σ, 2*n);

    syndrome = compute_syndrome(H,J,y);
    # println(syndrome)

    z = integer_solve(bottomSys, syndrome)
    η = compute_eta_overcomplete(Mh, Vt, syndrome, z )

    # println(compute_syndrome(H,symplectic_form(n),η-y))

    bp_result = run_belief_propagation!(tg, η, σ, 1);
    
    dec = hard_decision(bp_result, decision_H);
    # zp = integer_solve(bottomSys, dec);


    # println(mod.(Mh*J*bp_result,1))
    # println(mod.(Mh*J*(bp_result-η),1))


    # println("error: ",y )
    # println("eta: ",η )
    # println("bp_result: ",bp_result )
    # println("dec: ",dec)
    # println("potential correction: ",η - G*dec)

    # println("residual: ",y - (η - bp_result))

    # correction = compute_eta_overcomplete(Mh, Vt, syndrome, zp )
    # correction = H'*dec
    correction = η - G*dec


    commutator = ((y - correction )' * J * logical)
    success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)

    # residual = (y - correction)
    # success_condition = (Int64(round(sum([2*x^2 for x in residual[1:n] ])))%2==0)
    # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    # println("commutator: ",commutator)
    if success_condition
        return 1
    else
        current_best = norm(correction)
        for j in 1:length(correction)
            for k in [-2,-1,1,2]
                new_dec = dec[:]
                new_dec[j]= new_dec[j]+k
                new_candidate =  η - G*new_dec
                new_norm = norm(new_candidate)
                if new_norm<current_best
                    # println("found new best")
                    current_best = new_norm
                    correction = new_candidate
                end
            end
        end
        # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
        commutator = ((y - correction )' * J * logical)
        success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)
        if success_condition
            return 1
        else
            return 0
        end
    end
end




# th_pl = plot(yscale=:log10)
th_pl = plot()

samples = 1_000_000

for n in [3,11,19]

    println("doing n = $n")
    # println(n)
    J = symplectic_form(n)

    # initialize code
    code = GKP_Rep_Code(n)
    logical = vec(code.logical')
    # Initialize Tanner graph
    H = code.code;

    Mh, Vt, bottomSys = overcomplete_syndrome_preperation(H)

    G = inv(J*Mh')'
    decision_H = inv(G)
    sigmas = collect(0.37:0.01:0.45)
    # sigmas = [0.1]
    success_rates = Vector{Float64}(undef, length(sigmas))

    for l in 1:length(sigmas)

        σ = sigmas[l]

        # for _ in 1:samples
            
        # end

        success = @distributed (+) for i in 1:samples
            # println(n)
            sample_and_decode(code,H,J,G, decision_H,logical,σ, n, Mh, Vt, bottomSys)
        end
        

        sr =success/samples
        println("σ = $σ; success rate: ", sr)
        success_rates[l] = sr
    end

    plot!(th_pl,sigmas,success_rates, label="n = $n",markershape=:xcross)
end

title!("Decoder success probability")
xlabel!(L"$\sigma$")
ylabel!(L"$p_\mathrm{succ}$")
# display(th_pl)
savefig(th_pl, "repetition_code_thr_plot.pdf")

# readline()

# size(bp_result')
# size(code.logical')
# size(y)
# size(η)
# size((y + η  + bp_result)' )

# η
# y - η

# η  - correction

# y - correction


# code.logical'

# H*(y - correction )