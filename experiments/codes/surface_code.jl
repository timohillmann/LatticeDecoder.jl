function _local_search_order(params::Dict)
    return params[:local_search] ? append!([2], fill(1, params[:d] - 1)) : [0]
end

function surface_code_problem(params::Dict)
    code = GKP_Surface_Code(params[:d], false)
    M = code.code
    J = code.J

    H = -M * J
    G = J * inv(M)
    return QECProblem(H, G, _local_search_order(params))
end

function surface_code_css_problem(params::Dict)
    code = GKP_Surface_Code(params[:d], false)
    M = code.code

    half = size(M, 1) ÷ 2
    Mqq = M[1:half, 1:half]
    Mpp = M[(half + 1):end, (half + 1):end]

    Gqq = -inv(Mqq)
    Gpp = inv(Mpp)

    if params[:basis] == "X"
        return QECProblem(Mqq, Gpp, _local_search_order(params))
    elseif params[:basis] == "Z"
        return QECProblem(Mpp, Gqq, _local_search_order(params))
    else
        error("Invalid basis. Choose either 'X' or 'Z'.")
    end
end

function run_surface_code_experiment(params::Dict, n_samples::Int64)
    problem = surface_code_problem(params)
    params[:iterations] = size(problem.H, 2)
    return qec_experiment(; problem, n_samples, params), problem.H
end

function run_surface_code_css_experiment(params::Dict, n_samples::Int64)
    problem = surface_code_css_problem(params)
    params[:iterations] = size(problem.H, 2)
    return qec_experiment(; problem, n_samples, params), problem.H
end
