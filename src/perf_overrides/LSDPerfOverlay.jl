module LSDPerfOverlay

using LatticeDecoder

const LD = LatticeDecoder

include("lsd_workspace.jl")
include("lsd_kernels_optimized.jl")
include("bp_lsd_optimized_runner.jl")

export run_belief_propagation_lsd_optimized!
export activate_lsd_overlay!
export clear_lsd_workspaces!

end
