using Distributed
@everywhere using NPZ
@everywhere using LaTeXStrings
@everywhere using LLLplus
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




@everywhere function sample_and_decode(code::QuantumCode,H::AbstractMatrix,J::AbstractMatrix,G::AbstractMatrix, decision_H::AbstractMatrix,logical::AbstractVector,σ::Float64,n::Int64, Mperp_LLL::AbstractMatrix)
    # tg = initialize_tanner_graph(code);
    tg = initialize_tanner_graph(H);
    
    # println("new error")
    y= sample_error(σ, 2*n);
    y[(n+1):end].=0
    M = H*J
    # syndrome = compute_syndrome(M,J,y);
    syndrome = M*J*y
    for s in syndrome
        s = s - round(s)
    end
    # println(syndrome)

    # η = compute_eta(H, syndrome)
    η = Vector(inv(M*J) * syndrome)

    # reduce the error candidate η to the centered parallelepiped of the dual lattice
    # η = η - Mperp_LLL'*round.(inv(Mperp_LLL')*η)

    # println("is this a dual lattice point? ",round.(compute_syndrome(M,J,Mperp_LLL'*round.(inv(Mperp_LLL')*η)),digits=3))
    # println(round.(compute_syndrome(M,J,η-y),digits=3))

    bp_result = run_belief_propagation!(tg, η, σ/sqrt(2), 40); 
    
    # dec = hard_decision(bp_result, decision_H); 
    dec = round.(inv(decision_H)*bp_result)
    
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
    
    # dec = round.(inv(Mperp_LLL)'*η) # COMMENT THIS UNLESS YOU WANT ONLY LOCAL SEARCH
    correction = η - G*dec
    # println("the decoded vector ",round.(Int64,sqrt(2) *G*dec).%2)
    correction[n+1:end] .= 0 
    
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

    residual = y - correction
    # println("residual: ",residual)
    
    # commutator = (residual' * J * logical)
    # success_condition = ( abs(commutator[1]-round(commutator[1])) < 1e-3)
    success_condition = !is_logical_error(code,residual)
    # success_condition = !(norm(y)<norm(correction))

    # residual = (y - correction)
    # success_condition = (Int64(round(sum([2*x^2 for x in residual[1:n] ])))%2==0)
    # success_condition = (norm(y[1:n] - correction[1:n]) <1e-3)
    # println("commutator: ",commutator)
    if success_condition
        # println("success: ",round.(residual,digits=3))
        return 1
    end

    # println("fail: ",round.(residual,digits=3))
    
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

# th_pl = plot(yscale=:log10,yticks=[10^x for x in [-2.5,-2.25,-2,-1.75, -1.5,-1.25,-1,-0.75,-0.5,-0.25]],ylims=(10^(-2.5),10^(-0.25)))
th_pl = plot()


samples = 10_000

# for n in [5]

global jj = 1
for n in [3,7]
    println("doing n = $n")
    # println(n)
    J = symplectic_form(n)



    # initialize code
    code = GKP_Rep_Code(n, false,true)
    logical = vec(code.logical')
    display(logical')

    # Initialize Tanner graph
    # M is the GKP generator, 
    M = code.code
    # display(M)
    # H is the matrix used to construct the Tanner graph for BP
    H = -M * J 

    # we use the generator M to define the generator of the dual lattice
    Mperp = inv(J*M')
    # and we LLL reduce it 
    # (note that LLLPlus uses the column-convention: 
    # columns of the matrix generate the lattice - we use row-convention)
    B, _ = LLLplus.lll(Mperp',0.95)
    Mperp_LLL = B'

    # display(Mperp_LLL*J*M')

    # we can check that the determinant of the generator gives the correct logical dimension
    # println("determinant of reduced generator = $(det(M))")

    # the generator used in the classical algorithm to compute BP solution
    G = inv(H)
    decision_H = H
    
    # values for the noise strength (linear scale, lattice units)
    # sigmas = collect(0.1:0.1:0.7)
    sigmas = collect(0.5:0.1:2) ./sqrt(2*pi)
    # sigmas = [0.35]

    failure_rates = Vector{Float64}(undef, length(sigmas))

    # @everywhere tg = initialize_tanner_graph(H);

    for l in 1:length(sigmas)

        σ = sigmas[l]

        # for _ in 1:samples
            
        # end

        success = @distributed (+) for i in 1:samples
            # println(n)
            sample_and_decode(code,H,J,G, decision_H,logical,σ, n,Mperp_LLL)
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








title!("Decoder failure probability (reduced)")
xlabel!(L"$\sigma$")
ylabel!(L"$p_\mathrm{fail}$")
# display(th_pl)
# readline()

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