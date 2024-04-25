using LatticeAlgorithms
using Plots
using Counters
using LinearAlgebra


using Distributed
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


function logical_error(b::AbstractVector, Mperp::AbstractMatrix)
    commutators = Mperp * Ω_matrix(Mperp) * Mperp'*b
    return (any(commutators-round.(commutators) .>1e-3))
end

num_samples = 10_000

sigmas = range(0.15, 0.4, 11);
search_intervals = [sqrt(2)]  # range(0.5, 1.0, 31);
Ns = [3,13,23];
p = plot(xlabel="sigma", ylabel="Rate", title="Logical error rate for GKP repetition code")
for N in Ns
    rate = []
    rate_alt = []
    for e in search_intervals
        for σ in sigmas
            M = rep_rec(N)
            tg = initialize_tanner_graph(M)
            # Mperp = GKP_logical_operator_generator_canonical(M)
            Mperp = inv(Ω_matrix(M)*M')
            ξs = [σ * randn(2 * N) for _ in 1:num_samples]


            ys = []
            ys_alt = []
            for ξ in ξs
                y = run_belief_propagation!(tg, ξ, σ, N, e)
                push!(ys, copy(y))

                s = mod.(( sqrt(2π)* M * inv(Ω_matrix(M)) * ξ ), 2π );
                η = (-1/sqrt(2π))*inv(M*Ω_matrix(M)) * s 
                y_alt = run_belief_propagation!(tg, η, σ, N, e)
                push!(ys, copy(y))
                push!(ys_alt, copy(y_alt))
            end
            bs = [hard_decision(y, Mperp') for y in ys]
            push!(rate, count(x->logical_error(x, Mperp),bs)/num_samples)
            bs_alt = [hard_decision(y_alt, Mperp') for y_alt in ys_alt]
            push!(rate_alt, count(x->logical_error(x, Mperp),bs_alt)/num_samples)
            # us = [inv(transpose(sqrt(2π) * Mperp)) * y for y in ys]
            # us_alt = [inv(transpose(sqrt(2π) * Mperp)) * y_alt for y_alt in ys_alt]

            # logicals = [mod.(round.(Int, u[1:2]), 2) for u in us]
            # push!(rate, 1 - counter(logicals)[[0, 0]] / num_samples)
            
            # logicals_alt = [mod.(round.(Int, u[1:2]), 2) for u in us]
            # push!(rate_alt, 1 - counter(logicals_alt)[[0, 0]] / num_samples)
        end
        plot!(p, sigmas, rate, label="N = $N, interval = $e")
        plot!(p, sigmas, rate_alt, label="N_alt = $N, interval = $e",linestyle=:dash)
    end
    # println("rate: ", rate)
    # println("rate_alt: ", rate_alt)

end
# set axis log scale
plot!(p, yscale=:log10)

# plot horizontal line at 0.75
# plot!(p, [min(sigmas...), max(sigmas...)], [0.75, 0.75], label="0.75", lw=1, ls=:dash, color=:black)

display(p)
# N = 3
# σ = 0.8
# M = rep_rec(N)
# Mperp = GKP_logical_operator_generator_canonical(M)

# # this to force an additional logical operator on top of the error
# # b = zeros(Int64, 2 * N)
# # b[3] = 1
# # ξ0 = Mperp'*b
# # M* Ω_matrix(M) * ξ0

# # ξ0
# using LinearAlgebra
# det(M)
# det(Mperp)
# sqrt(abs(det(M)/det(Mperp)))

# ξ = σ * randn(2 * N)

# s = M * Ω_matrix(M) * ξ - round.(Int,M * Ω_matrix(M) * ξ  )
# η = inv(M * Ω_matrix(M)) * s


# println("integer if the two inputs to the decoder differ by a logical \n",inv(Mperp)'*(ξ - η))
# println("integer if the two inputs to the decoder differ by a logical \n",M*Ω_matrix(M) * (ξ - η))


# y = decode_rep_rec(sqrt(2π)* ξ) ./ sqrt(2π)
# println("integer? ", M * Ω_matrix(M) *y)
# println("half-integer? ", Mperp * Ω_matrix(M) *y)
# residual = ξ-y


# yp = decode_rep_rec(sqrt(2π)*η) ./ sqrt(2π)
# println("integer? ", M * Ω_matrix(M) *yp)
# println("half-integer? ", Mperp * Ω_matrix(M) *yp)
# residualp = ξ-η+yp

# println("integer if the two residuals differ by a logical ",inv(Mperp)'*(residual - residualp))
# println("integer if the two residuals differ by a logical ",M*Ω_matrix(M)*(residual - residualp))

# println("integer if the two residuals differ by a stabilizer ",inv(M)'*(residual - residualp))



# u = inv(Mperp)' * residual
# logical = mod.(round.(Int, u[1:2]), 2)
# up = inv(Mperp)' * residualp
# logicalp = mod.(round.(Int, up[1:2]), 2)

# Ω_matrix(M)*Ω_matrix(M)

########################################################
########################################################
### look from here


N = 3
σ = 0.1
M = rep_rec(N)
# Mperp = GKP_logical_operator_generator_canonical(M)
Mperp = inv(Ω_matrix(M)*M') # SEEMS TO WORK

ξ = σ * randn(2 * N);
println("error: ",ξ)


# using eq 54 and 56 from Mao Lin's paper
s = mod.(( sqrt(2π)* M * inv(Ω_matrix(M)) * ξ ), 2π );
# for e in s
#     if e>π
#         e = -2π + e
#     end
# end
# η = (-1/sqrt(2π))*(Ω_matrix(M)*Mperp)' * s  ;
η = (-1/sqrt(2π))*inv(M*Ω_matrix(M)) * s # USING THIS DEFINITION OF η
(ξ-η)
b=(-1/sqrt(2π))*inv(Mperp')*(ξ-η)
mod.(( -sqrt(2π)* M * Ω_matrix(M) * (ξ- η )), 2π )



println("all zeros if η has the same syndrome of ξ \n",mod.(( -sqrt(2π)* M * Ω_matrix(M) * η ), 2π )-s)



y = decode_rep_rec( ξ);
shortest_s_comp_vec = ξ-y;
println("shortest syndrome-compatible vector if decoding ξ directly\n")
println(shortest_s_comp_vec)
println("norm: ",norm(shortest_s_comp_vec))

yp = decode_rep_rec(η);
shortest_s_comp_vec_p =η-yp;
println("shortest syndrome-compatible vector if decoding η\n")
println(shortest_s_comp_vec_p)
println("norm: ",norm(shortest_s_comp_vec_p))


residualp = ξ-shortest_s_comp_vec_p; # see right after eq 57

new_s = ( -sqrt(2π)* M * Ω_matrix(M) * (residualp) ) ./ 2π ;

println("all integers if decoding η leads to stabilizer or logical error \n",new_s)

M * Ω_matrix(M) * Mperp'







tg = initialize_tanner_graph(M)
ybp = run_belief_propagation!(tg, ξ, σ, N)
ybp_p = run_belief_propagation!(tg, η, σ, N)
b = hard_decision(ybp, Mperp')
bp = hard_decision(ybp_p, Mperp')

Mperp * Ω_matrix(M) * Mperp'*b
Mperp * Ω_matrix(M) * Mperp'*bp




# us = [inv(transpose(sqrt(2π) * Mperp)) * y for y in ys]
# logicals = [mod.(round.(Int, u[1:2]), 2) for u in us]
# push!(rate, 1 - counter(logicals)[[0, 0]] / num_samples)



