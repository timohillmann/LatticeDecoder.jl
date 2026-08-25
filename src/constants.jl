# Centralized package-level constants and hard-coded defaults.

const LD = @__MODULE__

# Code constructors
const ACCEPTANCE_THRESHOLD = 1e-5
const CLASSICAL_LDLC_MAX_ATTEMPTS = 1000
const DEFAULT_SPARSE_CODE_MAX_ITERS = 500_000

# Gaussian arithmetic
const MIN_VAR::Float64 = 1e-10

# Local search
const DEFAULT_MAX_MATERIALIZED_LOCAL_SEARCH_CANDIDATES = 10_000
const SPHERE_DECODER_PRUNE_ATOL = 1e-15
const SPHERE_DECODER_OFFSET_CAP = 1000

# List sphere decoding
const LSD_DEFAULT_BETA = 3.5
const LSD_W_MIN = 0.95
const LSD_EPSILON = 1e-10
const LSD_MAX_ITER = 1000
const LSD_DEFAULT_MAX_CANDIDATES = 128
const LSD_TLS_ALLOCATIONS_KEY = :lsd_perf_overlay_allocations

# Backwards-compatible aliases used by the allocating/reference LSD path.
const W_MIN = LSD_W_MIN
const EPSILON = LSD_EPSILON
const MAX_ITER = LSD_MAX_ITER

# Belief propagation allocation defaults
const BP_DEFAULT_ALLOC_VARIANCE = 0.5

# Data collection output
const RESULTS_CSV_HEADER = "     shots,    errors,  discards, seconds,decoder,strong_id,json_metadata,custom_counts"
const HEADER = RESULTS_CSV_HEADER

function _validate_lsd_beta(lsd_beta::Float64)
    lsd_beta > 0 || throw(ArgumentError("LSD beta must be positive."))
    return nothing
end

function _validate_lsd_w_min(lsd_w_min::Float64)
    lsd_w_min >= 0 || throw(ArgumentError("LSD w_min must be nonnegative."))
    return nothing
end
