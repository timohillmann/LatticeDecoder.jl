using LatticeDecoder
using BenchmarkTools
d, n = 5, 128
code = classical_ldlc(d, n);

tg = initialize_tanner_graph(code);

tg.lsd_mem;

y = sample_error(0.15, n);

using LatticeDecoder: initialize_messages!, check_node_iterations!

vn = tg.var_nodes[1]
cn_message = vn.messages[1]
nb_idx = 1
mem = tg.lsd_mem[length(vn.neighbours)]

initialize_messages!(tg, y, 0.15)
check_node_iterations!(tg)

@benchmark LatticeDecoder.update_msg_vector!(mem, vn, nb_idx)

@benchmark LatticeDecoder.ListSphereDecodingInput!(mem)

@benchmark LatticeDecoder.simplified_lsd!(mem)

@benchmark LatticeDecoder.moment_matching!(cn_message, mem.candidate_gaussians[1:mem.found_messages])

@benchmark LatticeDecoder.update_candidate_gaussians!(mem)

@benchmark LatticeDecoder.initialize_messages!(tg, y, 0.15)

@benchmark LatticeDecoder.check_node_iterations!(tg)

@benchmark LatticeDecoder.variable_node_iterations_lsd_mem!(tg)

@benchmark LatticeDecoder.decision_step_lsd_mem!(tg)

@benchmark run_belief_propagation!(tg, y, 0.15, 15, "lsd")

