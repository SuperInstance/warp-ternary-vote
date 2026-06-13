# warp-ternary-vote

GPU **warp-level ternary voting** simulation — models 32-thread CUDA warps where each thread holds a ternary value {-1, 0, +1}, providing ballot, reduce, majority, and block-level consensus operations.

## Why It Matters

Modern GPUs execute in **warps** of 32 threads with warp-level intrinsics (`__ballot_sync`, `__shfl_down_sync`, `__all_sync`, `__any_sync`) that execute in ~4 clock cycles. These intrinsics are the fastest hardware mechanism for collective decisions among parallel agents.

This crate simulates that hardware model for ternary agents: each thread is an agent voting {-1 (reject), 0 (neutral), +1 (accept)}. The warp operations then map directly to agent consensus primitives — enabling 10,000-agent decisions in microseconds when deployed on real GPU hardware.

## How It Works

### CUDA Warp Model

A GPU **warp** is 32 threads executing in lockstep (SIMT). The key warp intrinsics map to this crate as:

| CUDA Intrinsic | Crate Method | Cost |
|---|---|---|
| `__ballot_sync(mask, pred)` | `Warp::ballot()` | 4 cycles |
| `__shfl_down_sync()` reduction | `Warp::reduce_sum()` | ~20 cycles |
| `__all_sync(mask, pred)` | `Warp::all_nonzero()` | 4 cycles |
| `__any_sync(mask, pred)` | `Warp::any_positive()` | 4 cycles |

### Ballot Counting

For each warp, the ballot counts the three ternary values:

```
ballot = { accept: |{i : v_i = +1}|,
           reject: |{i : v_i = −1}|,
           neutral: |{i : v_i =  0}| }
```

- **Complexity:** O(WARP_SIZE) = O(32) = O(1)
- accept + reject + neutral = 32 (conservation)

### Warp Reduce

Sum all ternary values in the warp:

```
reduce_sum = Σᵢ v_i,  v_i ∈ {-1, 0, +1}
```

- **Complexity:** O(WARP_SIZE)
- **Range:** [−32, +32]
- reduce_sum = 0: balanced (equal accept and reject)
- reduce_sum = +32: unanimous accept
- reduce_sum = −32: unanimous reject

### Majority Vote

```
majority(warp) = argmax(accept, reject, neutral)
```

Ties break toward 0 (neutral wins when no clear majority).

- **Complexity:** O(WARP_SIZE)

### Block-Level Consensus

A `WarpBlock` contains multiple warps. Block consensus uses a **threshold** rule:

```
decision = +1  if total_accept ≥ (N/2 + 1)
         = −1  if total_reject ≥ (N/2 + 1)
         =  0  otherwise
```

Where N = total threads across all warps.

**Hierarchical reduce:** Each warp reduces independently (log-step tree reduction on hardware), then warp results are summed:

```
block_sum = Σ_warps warp_reduce_sum
```

- **Complexity:** O(W × 32) where W = number of warps

## Quick Start

```rust
use warp_ternary_vote::*;

// Create a warp with 20 accept, 12 reject
let mut vals = [0i8; 32];
for i in 0..20 { vals[i] = 1; }
for i in 20..32 { vals[i] = -1; }
let warp = Warp::new(0, vals);

assert_eq!(warp.ballot().accept, 20);
assert_eq!(warp.ballot().reject, 12);
assert_eq!(warp.majority(), 1);  // accept wins
assert_eq!(warp.reduce_sum(), 8); // 20 - 12

// Block consensus: 8 uniform-accept warps
let block = WarpBlock::new(vec![Warp::uniform(0, 1); 8]);
let consensus = block.consensus();
assert_eq!(consensus.decision, 1);  // accept
assert_eq!(consensus.total_threads, 256);
```

## API

| Type | Purpose |
|---|---|
| `Warp` | 32-thread ternary vote unit |
| `WarpBallot` | Count of accept/reject/neutral in a warp |
| `WarpBlock` | Collection of warps for block-level consensus |
| `BlockConsensus` | Aggregated decision across all warps |

### Warp Methods
- `ballot()` → `WarpBallot` — count votes
- `reduce_sum()` → `i32` — sum all values
- `majority()` → `i8` — plurality winner
- `all_nonzero()` → `bool` — unanimous non-neutral
- `any_positive()` → `bool` — at least one accept

## Architecture Notes

The ballot structure directly instantiates the **γ + η = C** conservation law: within each warp, `accept + reject + neutral = 32`. The γ fraction (accept/32) and η fraction (reject/32) partition the active population, with the neutral fraction absorbing the remainder. Block consensus extends this: `total_accept + total_reject + total_neutral = total_threads`. The majority threshold (>50% of all threads) ensures that the γ fraction alone can drive a decision, but only when it exceeds the combined η + neutral opposition.

## References

- NVIDIA. *CUDA C++ Programming Guide.* §7.21: Warp Vote Functions.
- NVIDIA. *CUDA C++ Programming Guide.* §7.20: Warp Shuffle Functions.
- Hoefler, T. et al. (2013). *"MPI and Collective Communication on GPU Clusters."* — Hierarchical reduction.

## License

Apache-2.0
