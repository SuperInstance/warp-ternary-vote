# warp-ternary-vote

Experiment: GPU warp-level ternary voting simulation. 32 threads with {-1,0,+1} values, warp reduce, warp ballot, majority voting.

## Why This Matters

# warp-ternary-vote
GPU warp-level ternary voting simulation.
32 threads each hold a ternary value {-1, 0, +1}.
Simulates __ballot_sync(), warp reduce, and majority voting.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct Warp
pub fn new
pub fn uniform
pub fn ballot
pub fn reduce_sum
pub fn all_nonzero
pub fn any_positive
pub fn majority
pub struct WarpBallot
pub fn total
pub struct WarpBlock
pub fn new
```

## Usage

```toml
[dependencies]
warp-ternary-vote = "0.1.0"
```

```rust
use warp_ternary_vote::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/warp-ternary-vote.git
cd warp-ternary-vote
cargo test    # 7 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 7 |
| Lines of Rust | 189 |
| Public API | 15 items |

## License

Apache-2.0
