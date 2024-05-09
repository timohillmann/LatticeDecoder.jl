using Distributed
@everywhere using LaTeXStrings
# if nprocs()<6
#     addprocs(6-nprocs())
# end
# @everywhere using LatticeDecoder
@everywhere using LinearAlgebra
@everywhere using Plots
# using LatticeAlgorithms
@everywhere using LLLplus


if true
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/lattice_tools/overcomplete_syndrome.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/code_constructors/rep_codes.jl")

    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/utilities/utilities.jl")

    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians.jl")
    # include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/gaussians_log_weights_OLD.jl")
    # @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl")
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/parallel_bp_log_weight.jl")
    
    @everywhere include("/home/frarzani/Documents/QAT/research/LatticeDecoder.jl/src/bp_algorithms/list_sphere_decoder_log_weight.jl")
end

# include("../src/code_constructors/rep_codes.jl")


@everywhere function sample_and_decode(code::QuantumCode,H::AbstractMatrix,J::AbstractMatrix,G::AbstractMatrix, decision_H::AbstractMatrix,logical::AbstractVector,σ::Float64,n::Int64, Mh, Vt, bottomSys, Mperp_LLL::AbstractMatrix)
    # tg = initialize_tanner_graph(code);
    tg = initialize_tanner_graph(H);

    # println("new error")
    y= sample_error(σ, 2*n);
    # for j in (n+1):(2*n)
    #     y[j] = 0
    # end
    syndrome = compute_syndrome(H*J,J,y);
    # println(syndrome)
    
    z = integer_solve(bottomSys, syndrome)
    η = compute_eta_overcomplete(Mh, Vt, syndrome, z )

    # println("before reducing to parallelepiped length of η = ",norm(η))
    # reduce the error candidate η to the centered parallelepiped of the dual lattice
    η = η - Mperp_LLL'*round.(inv(Mperp_LLL)'*η) 
    # println("after reducing to parallelepiped length of η = ",norm(η))

    
    # println("measured syndrome = ", round.(syndrome,digits=3))
    # println("η syndrome = ", round.(compute_syndrome(H*J,J,η),digits=3))

    bp_result = run_belief_propagation!(tg, η, σ/1.5, 50);
    
    dec = hard_decision(bp_result, decision_H);
    # zp = integer_solve(bottomSys, dec);
    
    
    # println(mod.(Mh*J*bp_result,1))
    # println(mod.(Mh*J*(bp_result-η),1))
    
    
    # println("error: ", round.(y,digits = 3) )# UNCOMMENT FOR VERBOSE
    # println("eta: ",round.(η,digits = 3) )# UNCOMMENT FOR VERBOSE
    # println("bp_result: ",bp_result )
    # println("dec: ",dec)
    # if any(abs.(round.(syndrome-compute_syndrome(H*J,J,η),digits=3)).>0) # UNCOMMENT FOR VERBOSE
    #     println("diff syndrome: ",round.(syndrome-compute_syndrome(H*J,J,η),digits=3))
    # end
    # println("potential correction: ",round.(η - G*dec,digits=3))# UNCOMMENT FOR VERBOSE
    
    
    # correction = compute_eta_overcomplete(Mh, Vt, syndrome, zp )
    # correction = H'*dec
    correction = η - G*dec
    
    
    # println("residual: ",residual)
    current_best_norm = norm(correction)
    integer_vectors = npzread("examples/local_search/integer_vectors_$n.npz")["arr_0"]
    
    # for j in 1:(2*n)
    for j in 1:(size(integer_vectors)[1])
        # for k in (-3):3
        new_dec = dec[:]
        # new_dec[j]= new_dec[j]+k
        new_dec = new_dec + integer_vectors[j,:]
        new_candidate =  η - G*new_dec
        # new_candidate[n+1:end] .= 0
        new_norm = norm(new_candidate)
        if new_norm<current_best_norm
            # println("found new best")
            current_best_norm = new_norm
            correction = new_candidate
        end
        # end
    end
    #     # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    #     commutator = ((y - correction )' * J * logical)
    #     success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)
    #     # residual = y - correction
    #     if success_condition
    # residual = (y - correction)
    # success_condition = (Int64(round(sum([2*x^2 for x in residual[1:n] ])))%2==0)
    # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    # println("commutator: ",commutator)
    
    # println("revised correction: ",round.(correction,digits=3)) # UNCOMMENT FOR VERBOSE

    
    residual = y - correction
    # println("residual: ",round.(residual,digits=3))# UNCOMMENT FOR VERBOSE

    # println("error larger than correction? $(norm(y)>norm(correction))") # UNCOMMENT FOR VERBOSE

    # check if resitual error is a (possibly trivial) lgical operator
    if (norm(Mh*J*residual - round.(Mh*J*residual))>1e-6)
        println("non-logical residual?",display(residual'))
    end

    # display(round.(residual',digits=3))

    success_condition = !is_logical_error(code,residual)
    # success_condition = (norm(y)>=norm(correction))
    if success_condition
        # println("success!")# UNCOMMENT FOR VERBOSE
        # println("\n")# UNCOMMENT FOR VERBOSE
        return 1
    end
    # println("error: $(round.(y,digits=4))");
    # println("correction: $(round.(correction,digits=4))");
    # println("failed with residual:  $(round.(residual,digits=4))\n")
    # println("\n")# UNCOMMENT FOR VERBOSE
    return 0
end


matching_results = Dict(
    "5" => [0.0035, 0.0221, 0.067, 0.127, 0.1856, 0.2684, 0.3302, 0.3819, 0.4273, 0.4449, 0.4639, 0.4712, 0.4984, 0.5187, 0.4965, 0.5208],
    "13" => [0.0, 0.0005, 0.0084, 0.0343, 0.0927, 0.1548, 0.243, 0.3121, 0.3749, 0.4192, 0.4603, 0.4869, 0.5005, 0.5248, 0.55, 0.5547],
    "7" => [0.0012, 0.0098, 0.0341, 0.0914, 0.1564, 0.232, 0.2994, 0.3615, 0.4036, 0.4385, 0.4709, 0.476, 0.5083, 0.5072, 0.5122, 0.527],
    "11" => [0.0, 0.0012, 0.0145, 0.0465, 0.1082, 0.1826, 0.2576, 0.3282, 0.3857, 0.4251, 0.4642, 0.4867, 0.4955, 0.5096, 0.5292, 0.541],
    "9" => [0.0005, 0.0046, 0.0228, 0.0598, 0.1289, 0.2052, 0.2778, 0.3374, 0.3912, 0.43, 0.46, 0.4766, 0.5039, 0.5185, 0.527, 0.5333],
    "sigmas" => [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "3" => [0.0156, 0.0508, 0.1045, 0.1738, 0.2447, 0.3065, 0.3597, 0.394, 0.4266, 0.4591, 0.4815, 0.4888, 0.4953, 0.5047, 0.5048, 0.5065]
)



th_pl = plot(yscale=:log10,yticks=[10^x for x in  [-5,-4,-3,-2,-2.25,-2,-1.75, -1.5,-1.25,-1,-0.75,-0.5,-0.25]],ylims=(10^(-5),10^(-0.25)))
th_pl = plot()

samples = 10

global jj = 1
for n in [3]
# for n in [3,9,15]
    println("doing n = $n")
    # println(n)
    J = symplectic_form(n)



    # initialize code
    code = GKP_Rep_Code(n)
    logical = vec(code.logical')
    # println("logical: ", round.(logical',digits=3),"\n")

    # Initialize Tanner graph
    # M is the GKP generator, 
    M = code.code;
    # println("M: \n")
    # display(round.(M,digits = 4))
    # println("\n")
    # H is the matrix used to c, while in factonstruct the Tanner graph for BP
    H = -M * J

    # we prepare the system to compute an error candidate compatible
    # with the overcomplete syndrome obtained from measuring stabilizers in the rows of M
    Mh, Vt, bottomSys = overcomplete_syndrome_preperation(M)
    
    # we use the Hermite reduced Mh to define the generator of the dual lattice
    Mperp = -inv(J*Mh')

    # and we LLL reduce it 
    # (note that LLLPlus uses the column-convention: 
    # columns of the matrix generate the lattice - we use row-convention)
    B, _ = LLLplus.lll(Mperp',0.98)
    Mperp_LLL = B'

    display(Mperp_LLL)
    # we can check that the determinant of the generator gives the correct logical dimension
    # println("determinant of reduced generator = $(det(Mh))")
    
    
    
    # in the case of an overcomplete syndrome we need to calculate
    # the decision_H from an invertible matrix so that we also 
    # can calculate the generator matrix for the classical code generation
    decision_H = -Mh * J
    
    
    
    # the generator used in the classical algorithm to compute BP solution
    G = inv(decision_H)
    
    
    # values for the noise strength (linear scale, lattice units)
    # sigmas = collect(0.2:0.05:0.55)
    sigmas = collect(0.5:0.1:2) ./sqrt(2*pi)
    # sigmas = [0.55]
    
    failure_rates = Vector{Float64}(undef, length(sigmas))
    
    # @everywhere tg = initialize_tanner_graph(H);
    for l in 1:length(sigmas)

        σ = sigmas[l]

        
        # success = @distributed (+) for i in 1:samples
        #     sample_and_decode(code,H,J,G, decision_H,logical,σ, n, Mh, Vt, bottomSys,Mperp_LLL)
        #     # println("decision H is now of size $(size(decision_H))")
        # end
        
        success = 0
        for i in 1:samples
            success += sample_and_decode(code,H,J,G, decision_H,logical,σ, n, Mh, Vt, bottomSys,Mperp_LLL)
        end

        fr =1-success/samples
        println("σ = $σ; failure rate: ", fr)
        failure_rates[l] = fr
    end

    plot!(th_pl,sigmas,failure_rates, label="n = $n",markershape=:circle)

    # Access the series objects
    series = th_pl.series_list

    # Get the color of the second series
    color1 = series[jj][:seriescolor]
    plot!(th_pl,sigmas,matching_results["$n"], label="MWPM, n = $n",markershape=:xcross,color = color1)
    global jj+=2
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