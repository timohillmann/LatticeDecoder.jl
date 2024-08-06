using Distributed
@everywhere using LaTeXStrings
if nprocs()<6
    addprocs(6-nprocs())
end
@everywhere using LinearAlgebra
@everywhere using Plots
# using LatticeAlgorithms
@everywhere using LLLplus
@everywhere using Nemo
@everywhere using DelimitedFiles
@everywhere using LatticeDecoder
@everywhere using Colors


if true
    @everywhere include(realpath(dirname(@__FILE__))*"/../src/code_constructors/rep_codes.jl")

    # @everywhere include(realpath(dirname(@__FILE__))*"/../src/utilities/utilities.jl")

    # @everywhere include(realpath(dirname(@__FILE__))*"/src/bp_algorithms/gaussians.jl")
    # include(realpath(dirname(@__FILE__))*"/src/bp_algorithms/gaussians_log_weights_OLD.jl")
    # @everywhere include(realpath(dirname(@__FILE__))*"/src/bp_algorithms/parallel_bp.jl")
    @everywhere include(realpath(dirname(@__FILE__))*"/../src/bp_algorithms/parallel_bp_log_weight.jl")
    
    @everywhere include(realpath(dirname(@__FILE__))*"/../src/bp_algorithms/list_sphere_decoder_log_weight.jl")
end

@everywhere function sample_and_decode(M::AbstractMatrix, H::AbstractMatrix,G::AbstractMatrix, J::AbstractMatrix ,logical::AbstractMatrix,σ::Float64,n::Int)
    # tg = initialize_tanner_graph(code);
    tg = initialize_tanner_graph(H);
    
    # println("new error")
    y= sample_error(σ, 2*n);

    syndrome = M*Int64.(J)*y
    # println(syndrome)

    η = Vector(inv(H) * syndrome)


    # reduce the error candidate η to the centered parallelepiped of the dual lattice

    # println(round.(compute_syndrome(M,J,η-y),digits=3))

    bp_result = run_belief_propagation!(tg, η, 2*σ, 80); 
    # bp_result = η
    
    # dec = hard_decision(bp_result, decision_H); 
    dec = round.(Int64,inv(H)*bp_result)
    
    # println(mod.(Mh*J*bp_result,1))
    # println(mod.(Mh*J*(bp_result-η),1))
    
    
    # println("error: ",y )
    # println("eta: ",η )
    # println("bp_result: ",bp_result )
    # println("dec: ",dec)
    # println("potential correction: ",η - G*dec)
    
    # println("residual: ",y - (η - bp_result))
    
    correction = η - G*dec
    # println("the decoded vector ",round.(Int64,sqrt(2) *G*dec).%2)
    
    residual = y + correction
    # println("residual: \n", residual)
    # println("error: \n", y)

    # println("is the residual zero syndrome?: \n", M*J*residual)
    # residual = round.(BigInt,residual)
    # println("residual: ",residual)
    # display(J)
    # J = BigInt.(J)
    # println("type of logical: ",typeof(logical))
    # println("type of J: ",typeof(J))
    # println("type of residual: ",typeof(residual))
    
    commutator = (logical * J * residual ) # assuming logical is a matrix whose rows are the logical displacement vectors
    # println("commutator: \n",commutator,"\n")
    # println("commutator fractional part: " ,commutator - round.(Int64,commutator),"\n")
    # println("even/odd : ",(logical * J * residual )%2,"\n")
    

    # success_condition = all(commutator - round.(BigInt,commutator) .≈ 0)

    success_condition = all(abs.(commutator - round.(Int64,commutator) ) .<1e-3 )
    if success_condition
        # println("success: ",round.(residual,digits=3))
        return 1
    end

    # println("fail: ",round.(residual,digits=3))
    
    return 0
end


#######################################################
#######################################################
samples = 1000
# values for the noise strength (linear scale, lattice units)
# sigmas = collect(0.05:0.05:0.3)
sigmas = collect(0.1:0.1:1) ./sqrt(2*pi)
# sigmas = [0.01]

th_pl = plot()
title!("Decoder failure probability")
xlabel!(L"$\sigma$")
ylabel!(L"$p_\mathrm{fail}$")


#######################################################
#######################################################

println("computing single-mode error rates")
failure_rates = Vector{Float64}(undef, length(sigmas))
l = 1
for σ in sigmas
    success = 0
    for j in 1:1_000_000
        y= sample_error(σ, 2);
        # println(y)
        # syndrome = mod.(y, 1/sqrt(2));
        # println(syndrome)
        # residual = y-syndrome;
        # println(residual)
        # println(all(mod.(residual, sqrt(2)) .==0))
        residual = round.(Int64,sqrt(2)*y)./sqrt(2)

        success += all(mod.(residual, sqrt(2)) .≈ 0)
    end
    fr =1-success/1_000_000
    println("σ = $σ; failure rate: ", fr)
    failure_rates[l] = fr
    l+=1
end
plot!(th_pl,sigmas,failure_rates, label="single mode", markershape=:square,linecolor=:black, markercolor=:black)

#######################################################
#######################################################

# file_names = readlines("examples/ldlc_gens/qubit_containing_2.txt");#[1:1]


# file_names = readlines("examples/ldlc_gens/qubit_containing.txt")
file_names = filter(file-> !occursin("_A.npy",file), readdir("examples/ldlc_gens/large/"))


nmin =50
nmax =500
# file_names[1]

cgrad = range(colorant"red", stop=colorant"blue", length=(nmax-nmin+1));

# file_names[1:5]

for file_name in file_names

    # Define the regular expression pattern
    pattern = r"gkpldlc_(.*?).npy"

    # Extract the matched substring
    match_result = match(pattern, file_name)

    if isnothing(match_result)
        println("No match found")
    else
        substring = match_result.captures[1]
        println("Extracted substring: $substring")
    end

    n , index = parse.(Int64,split(substring, "_"))
    println(n," ",index)
    # exit()
    # println(n)
    # J = BigInt.(symplectic_form(n))
    J = symplectic_form(n)
    
    # M is the GKP generator, 
    M = npzread("examples/ldlc_gens/large/gkpldlc_$(n)_$(index).npy")
    # display(M)

    # M_LLL = Nemo.lll(Nemo.matrix(ZZ,M)) 
    Mperp = inv(J*M')
    # logical_ext_LLL = Nemo.lll(Nemo.matrix(ZZ,twice_logical_ext)) 

    println("starting LLL of Mperp")
    B, _ = LLLplus.lll(Mperp',0.75)
    println("done")
    # B, _ = LLLplus.hkz(Mperp',)
    Mperp_LLL = B'
    # we take this as generator for the LDLC algorithm, thus with column-convention
    G = Mperp_LLL'
    # G = Mperp
    logical_to_input = G
    
    
    # in the case of an overcomplete syndrome we need to calculate
    # the decision_H from an invertible matrix so that we also 
    # can calculate the generator matrix for the classical code generation
    decision_H = inv(G)
    # we also LLL reduce this
    println("starting LLL of decision_H")
    B, _ = LLLplus.lll(decision_H',0.98)
    println("done")
    # B, _ = LLLplus.hkz(decision_H')
    # decision_H = B'



    # display(unique(logical_to_input*J*M'))
    # display(unique(logical_to_input*J*Mcan'-round.(BigInt,logical_to_input*J*Mcan')))
    # display(unique(logical_to_input*J*logical_to_input'))


    # H is the matrix used to construct the Tanner graph for BP
    H = -M * Int64.(J) 
    
    # we use the generator M to define the generator of the dual lattice
    
    # the generator used in the classical algorithm to compute BP solution
    G = inv(H)
    # println("type of G: ", typeof(G))


    
    failure_rates = Vector{Float64}(undef, length(sigmas))
    
    # @everywhere tg = initialize_tanner_graph(H);
    
    println("doing n = $n, index = $index")
    for l in 1:length(sigmas)
        
        σ = sigmas[l]
        
        # for _ in 1:samples
        
        # end
        success = @distributed (+) for i in 1:samples
            # println(n)
            sample_and_decode(M, H, G, Int64.(J), logical_to_input, σ, n)
        end
        

        fr =1-success/samples
        println("σ = $σ; failure rate: ", fr)
        failure_rates[l] = fr
    end

    # plot!(th_pl,sigmas,failure_rates, label="n = $n, index = $index",markershape=:circle,linecolor=cgrad[n-14], markercolor=cgrad[n-14])
    plot!(th_pl,sigmas,failure_rates, label="", markershape=:circle,linecolor=cgrad[n-nmin+1], markercolor=cgrad[n-nmin+1])
    display(th_pl)

    # Access the series objects
    # series = th_pl.series_list

    # Get the color of the second series
    # color1 = series[jj][:seriescolor]
    # plot!(th_pl,sigmas,matching_results["$n"], label="MWPM, n = $n",markershape=:xcross,color = color1)
    # global jj+=2
end





display(th_pl)
# readline()

savefig(th_pl, "GKPLDLC_large_thr_plot.pdf")
exit()

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




n , index = 15,3
println(n," ",index)
# exit()
# println(n)
J = BigInt.(symplectic_form(n))

# M is the GKP generator, 
M = npzread("examples/ldlc_gens/gkpldlc_$(n)_$(index).npy")
# display(M)
Mcan = readdlm("examples/ldlc_gens/gkpldlc_$(n)_$(index)_canonical_M.npy.txt", ' ', BigInt) # Read space-delimited integers into matrix A
Mcan = BigInt.(Mcan*M)

logical = 0
for j in 1:(n)
        if abs(Mcan[j,:]'*J*Mcan[j+n,:]) ≈ 2
            logical = Mcan[[j,j+n],:]
            println("zeros if twice the logical ops have even symp prod with each stabilizer")
            display(Mcan[[j,j+n],:]*J*M'.%2)
        end
end

twice_logical_ext = vcat(2*M, logical)

using NormalForms
# display(logical_ext)

using Nemo
logical_ext_LLL = Nemo.lll(Nemo.matrix(ZZ,twice_logical_ext)) 
# logical_ext_LLL = (LLLplus.lll(logical_ext'))' # throws errors 

unique(logical_ext_LLL)

logical_to_input = Matrix{Int64}(logical_ext_LLL)./2

unique(logical_to_input*J*(M'))
unique(logical_to_input*J*(logical_to_input'))

# unique((Matrix{BigInt}(logical_ext_LLL))*J*(2*M)')

# unique((Matrix{BigInt}(logical_ext_LLL))*J*twice_logical_ext').%2