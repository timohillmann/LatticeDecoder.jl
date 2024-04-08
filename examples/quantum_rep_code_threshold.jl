using Distributed
using LaTeXStrings
# if nprocs()<6
#     addprocs(6-nprocs())
# end
# @everywhere using LatticeDecoder
@everywhere using LinearAlgebra
using Plots
# using LatticeAlgorithms


if true
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/lattice_tools/overcomplete_syndrome.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/code_constructors/rep_codes.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/utilities/utilities.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians.jl")
    # include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians_log_weights_OLD.jl")
    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl")
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp_log_weight.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl")
end

# include("../src/code_constructors/rep_codes.jl")


@everywhere function sample_and_decode(code::QuantumCode,H::AbstractMatrix,J::AbstractMatrix,G::AbstractMatrix, decision_H::AbstractMatrix,logical::AbstractVector,σ::Float64,n::Int64, Mh, Vt, bottomSys)
    # tg = initialize_tanner_graph(code);
    tg = initialize_tanner_graph(H);

    # println("new error")
    y= sample_error(σ, 2*n);
    for j in (n+1):(2*n)
        y[j] = 0
    end
    syndrome = compute_syndrome(H*J,J,y);
    # println(syndrome)
    
    z = integer_solve(bottomSys, syndrome)
    η = compute_eta_overcomplete(Mh, Vt, syndrome, z )
    
    # println(compute_syndrome(H*J,J,η))
    # println(syndrome,"\n")
    
    bp_result = run_belief_propagation!(tg, η, σ/2, 20);
    
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

    # check if resitual error is a (possibly trivial) lgical operator
    if (norm(Mh*J*residual - round.(Mh*J*residual))>1e-9)
        display(residual')
    end
    # println("residual: ",residual)
    # commutator = (residual' * J * logical)
    # success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)
    success_condition = !is_logical_error(code,residual)
    # residual = (y - correction)
    # success_condition = (Int64(round(sum([2*x^2 for x in residual[1:n] ])))%2==0)
    # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    # println("commutator: ",commutator)
    if success_condition
        # if !is_logical_error(code, residual)
        # println("diff syndrome: ",round.(syndrome-compute_syndrome(H*J,J,correction),digits=3))
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
    # println("error: $(round.(y,digits=4))");
    # println("correction: $(round.(correction,digits=4))");
    # println("failed with residual:  $(round.(residual,digits=4))\n")
    return 0
end




th_pl = plot(yscale=:log10,yticks=[10^x for x in [-1.75, -1.5,-1.25,-1,-0.75,-0.5,-0.25]],ylims=(10^(-1.75),10^(-0.25)))
# th_pl = plot()

samples = 500_000

# for n in [3,5,7]
for n in [3,9,15]
    println("doing n = $n")
    # println(n)
    J = symplectic_form(n)



    # initialize code
    code = GKP_Rep_Code(n)
    logical = vec(code.logical')
    println("logical: ", logical,"\n")

    # Initialize Tanner graph
    # M is the GKP generator, 
    M = code.code;
    println("M: \n")
    display(round.(M,digits = 4))
    println("\n")
    # H is the matrix used to c, while in factonstruct the Tanner graph for BP
    H = -M * J

    # we prepare the system to compute an error candidate compatible
    # with the overcomplete syndrome obtained from measuring stabilizers in the rows of M
    Mh, Vt, bottomSys = overcomplete_syndrome_preperation(M)

    # we can check that the determinant of the generator gives the correct logical dimension
    # println("determinant of reduced generator = $(det(Mh))")
    
    # in the case of an overcomplete syndrome we need to calculate
    # the decision_H from an invertible matrix so that we also 
    # can calculate the generator matrix for the classical code generation
    decision_H = -Mh * J

    # the generator used in the classical algorithm to compute BP solution
    G = inv(decision_H)
    
    # values for the noise strength (linear scale, lattice units)
    sigmas = collect(0.2:0.05:0.55)
    # sigmas = [0.35]

    failure_rates = Vector{Float64}(undef, length(sigmas))

    for l in 1:length(sigmas)

        σ = sigmas[l]

        success = @distributed (+) for i in 1:samples
            # println("decision H is now of size $(size(decision_H))")
            sample_and_decode(code,H,J,G, decision_H,logical,σ, n, Mh, Vt, bottomSys)
        end
        

        fr =1-success/samples
        println("σ = $σ; failure rate: ", fr)
        failure_rates[l] = fr
    end

    plot!(th_pl,sigmas,failure_rates, label="n = $n",markershape=:xcross)
end

title!("Decoder failure probability (overcomplete)")
xlabel!(L"$\sigma$")
ylabel!(L"$p_\mathrm{fail}$")
# display(th_pl)
# readline()

savefig(th_pl, "repetition_code_overcomplete_thr_plot.pdf")

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