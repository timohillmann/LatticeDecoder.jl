
# using JuMP
# using Gurobi
# using LinearAlgebra

# using LatticeDecoder

# function l1_minimize(b::AbstractVector, A::AbstractArray)
#     m, n = size(A)
#     model = Model(Gurobi.Optimizer)

#     @variable(model, x[1:n])
#     @variable(model, t[1:m] >= 0)

#     @objective(model, Min, sum(t[i] for i in 1:m))

#     for i in 1:m
#         # @constraint(model, dot(A[i, :], x) - b[i] >= t[i])
#         # @constraint(model, dot(A[i, :], x) - b[i] <= -t[i])
#         @constraint(model, t[i] >= dot(A[i, :], x) - b[i])
#         @constraint(model, t[i] >= -dot(A[i, :], x) - b[i])
#     end

#     optimize!(model)

#     return value.(x)
# end

# # Parameters
# n = 256;
# d = 5;
# H = classical_ldlc(d, n, true);

# # Initialize Tanner graph

# tg = initialize_tanner_graph(H);

# σ = 0.20;

# b = zeros(Int64, size(H, 1));
# b[1] = 0;
# b[2] = 0;

# G = generator_matrix(H);
# y = encode(b, G);

# y .+= sample_error(σ, n);

# serial_bp_result = run_serial_belief_propagation!(tg, y, σ, 10);
# serial_dec = hard_decision(serial_bp_result, H);

# bp_result = run_belief_propagation!(tg, y, σ, 10);
# dec = hard_decision(bp_result, H);

# println("Number of symbol errors: ", count_symbol_errors(dec, b))
# println("Number of symbol errors (serial): ", count_symbol_errors(serial_dec, b))

# # check that bp_results approximately fulfill the parity check equations
# println(round.(Int64, H * bp_result) .% 1)

# # decode by linear programml
# lp_result = l1_minimize(y, H)
# dec_lp = hard_decision(lp_result, H);

# bp_result
# serial_bp_result
# lp_result

# println("Number of symbol errors (lp): ", count_symbol_errors(dec_lp, b))
# println("Number of symbol errors (lp): ", count_symbol_errors(dec_lp, b))
# println("Number of symbol errors (lp): ", count_symbol_errors(dec_lp, b))
