using Distributed
using LaTeXStrings
if nprocs()<6
    addprocs(6-nprocs())
end
# @everywhere using LatticeDecoder
@everywhere using LinearAlgebra
using Plots
# using LatticeAlgorithms


if true
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/lattice_tools/overcomplete_syndrome.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/code_constructors/rep_codes.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/utilities/utilities.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/lattice_tools/concatenated_code_reduction.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians.jl")
    # include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians_log_weights_OLD.jl")
    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl")
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp_log_weight.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl")
end

# include("../src/code_constructors/rep_codes.jl")

@everywhere function greedy_reduction(pcm::AbstractMatrix)
    inds = [1]
    m, n = size(pcm)
    for i in 2:m
        if rank(pcm[inds, :]) < rank(pcm[[inds; i], :])
            push!(inds, i)
        end
    end

    return pcm[inds, :]
end




@everywhere function sample_and_decode(code::QuantumCode,H::AbstractMatrix,J::AbstractMatrix,G::AbstractMatrix, decision_H::AbstractMatrix,logical::AbstractVector,σ::Float64,n::Int64)
    # tg = initialize_tanner_graph(code);
    tg = initialize_tanner_graph(H);
    
    # println("new error")
    y= sample_error(σ, 2*n);

    M = H*J
    syndrome = compute_syndrome(M,J,y);
    # println(syndrome)

    # η = compute_eta(H, syndrome)
    η = -G * syndrome

    # println(compute_syndrome(H,symplectic_form(n),η-y))

    bp_result = run_belief_propagation!(tg, η, σ, 30);
    
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

    residual = y - correction
    # println("residual: ",residual)
    commutator = (residual' * J * logical)
    success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)

    # residual = (y - correction)
    # success_condition = (Int64(round(sum([2*x^2 for x in residual[1:n] ])))%2==0)
    # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    # println("commutator: ",commutator)
    if success_condition
    # if !is_logical_error(code, residual)
        return 1
    # else
    #     current_best = norm(correction)
    #     for j in 1:length(correction)
    #         for k in [-2,-1,1,2]
    #             new_dec = dec[:]
    #             new_dec[j]= new_dec[j]+k
    #             new_candidate =  η - G*new_dec
    #             new_norm = norm(new_candidate)
    #             if new_norm<current_best
    #                 # println("found new best")
    #                 current_best = new_norm
    #                 correction = new_candidate
    #             end
    #         end
    #     end
    #     # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    #     commutator = ((y - correction )' * J * logical)
    #     success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)
    #     # residual = y - correction
    #     if success_condition
    #     # if !is_logical_error(code, residual)
    #         return 1
    #     end
    end

    return 0
end




th_pl = plot(yscale=:log10)
# th_pl = plot()

samples = 500_000

# for n in [3]
for n in [3,7,11,15]
    println("doing n = $n")
    # println(n)
    J = symplectic_form(n)



    # initialize code
    code = GKP_Rep_Code(n, false,true)
    M_red = code.code

    logical = vec(code.logical')
    # Initialize Tanner graph
    # println("determinant of reduced generator = $(det(M_red))")
    # println("generator: \n", M_red)

    H = -M_red * J 
    # Mh, Vt, bottomSys = overcomplete_syndrome_preperation(H)

    G = inv(H)
    decision_H = H
    
    sigmas = collect(0.2:0.05:0.55)
    # sigmas = [0.35]

    failure_rates = Vector{Float64}(undef, length(sigmas))

    for l in 1:length(sigmas)

        σ = sigmas[l]

        # for _ in 1:samples
            
        # end

        success = @distributed (+) for i in 1:samples
            # println(n)
            sample_and_decode(code,H,J,G, decision_H,logical,σ, n)
        end
        

        fr =1-success/samples
        println("σ = $σ; failure rate: ", fr)
        failure_rates[l] = fr
    end

    plot!(th_pl,sigmas,failure_rates, label="n = $n",markershape=:xcross)
end

title!("Decoder failure probability")
xlabel!(L"$\sigma$")
ylabel!(L"$p_\mathrm{fail}$")
display(th_pl)
readline()

savefig(th_pl, "repetition_code_thr_plot.pdf")

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