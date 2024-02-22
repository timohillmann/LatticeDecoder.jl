"""
    ec_experiment(H, σ, max_iter, samples)

Run an error correction experiment on a parity-check matrix `H` with a Gaussian error vector of standard deviation `σ`.
The experiment is repeated `samples` times and the average number of bit errors, i.e., the bit error rate, is returned.
"""
function ec_experiment(H, σ, max_iter, samples)
    tg = initialize_tanner_graph(H)
    errors = 0

    for i = 1:samples
        y = sample_error(σ, size(H, 2))
        bp_result = run_belief_propagation!(tg, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        errors += count_bit_errors(dec)
    end

    return errors / samples
end