//! # warp-ternary-vote
//!
//! GPU warp-level ternary voting simulation.
//! 32 threads each hold a ternary value {-1, 0, +1}.
//! Simulates __ballot_sync(), warp reduce, and majority voting.

const WARP_SIZE: usize = 32;

/// A GPU warp — 32 threads each holding a ternary value.
#[derive(Debug, Clone)]
pub struct Warp {
    pub values: [i8; WARP_SIZE],
    pub id: u32,
}

impl Warp {
    pub fn new(id: u32, values: [i8; WARP_SIZE]) -> Self {
        Self { values, id }
    }

    pub fn uniform(id: u32, val: i8) -> Self {
        Self { values: [val; WARP_SIZE], id }
    }

    /// Warp ballot: count accept/reject/neutral votes.
    /// Maps to __ballot_sync(mask, predicate) in real CUDA.
    pub fn ballot(&self) -> WarpBallot {
        let accept = self.values.iter().filter(|&&v| v == 1).count();
        let reject = self.values.iter().filter(|&&v| v == -1).count();
        let neutral = self.values.iter().filter(|&&v| v == 0).count();
        WarpBallot { accept, reject, neutral, warp_id: self.id }
    }

    /// Warp reduce: sum all ternary values.
    /// Maps to __shfl_down_sync() reduction in real CUDA.
    pub fn reduce_sum(&self) -> i32 {
        self.values.iter().map(|&v| v as i32).sum()
    }

    /// Warp all: check if all threads have non-zero value.
    /// Maps to __all_sync(mask, predicate).
    pub fn all_nonzero(&self) -> bool {
        self.values.iter().all(|&v| v != 0)
    }

    /// Warp any: check if any thread has positive value.
    pub fn any_positive(&self) -> bool {
        self.values.iter().any(|&v| v > 0)
    }

    /// Majority vote: the value held by the most threads.
    pub fn majority(&self) -> i8 {
        let ballot = self.ballot();
        if ballot.accept > ballot.reject && ballot.accept > ballot.neutral { 1 }
        else if ballot.reject > ballot.accept { -1 }
        else { 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpBallot {
    pub accept: usize,
    pub reject: usize,
    pub neutral: usize,
    pub warp_id: u32,
}

impl WarpBallot {
    pub fn total(&self) -> usize { self.accept + self.reject + self.neutral }
}

/// A block of warps (e.g., 32 warps = 1024 threads).
pub struct WarpBlock {
    pub warps: Vec<Warp>,
}

impl WarpBlock {
    pub fn new(warps: Vec<Warp>) -> Self { Self { warps } }

    /// Block-level consensus: all warps vote, then aggregate.
    pub fn consensus(&self) -> BlockConsensus {
        let ballots: Vec<WarpBallot> = self.warps.iter().map(|w| w.ballot()).collect();
        let total_accept: usize = ballots.iter().map(|b| b.accept).sum();
        let total_reject: usize = ballots.iter().map(|b| b.reject).sum();
        let total_neutral: usize = ballots.iter().map(|b| b.neutral).sum();

        let threshold = (self.warps.len() * WARP_SIZE) / 2 + 1;
        let decision = if total_accept >= threshold { 1 }
            else if total_reject >= threshold { -1 }
            else { 0 };

        BlockConsensus {
            warps: self.warps.len(),
            total_threads: self.warps.len() * WARP_SIZE,
            total_accept, total_reject, total_neutral,
            decision,
        }
    }

    /// Warp-level reduce then block-level aggregate.
    pub fn hierarchical_reduce(&self) -> i32 {
        self.warps.iter().map(|w| w.reduce_sum()).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockConsensus {
    pub warps: usize,
    pub total_threads: usize,
    pub total_accept: usize,
    pub total_reject: usize,
    pub total_neutral: usize,
    pub decision: i8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_accept() {
        let warp = Warp::uniform(0, 1);
        assert_eq!(warp.ballot().accept, WARP_SIZE);
        assert_eq!(warp.reduce_sum(), WARP_SIZE as i32);
    }

    #[test]
    fn test_mixed_ballot() {
        let mut vals = [0i8; WARP_SIZE];
        for i in 0..10 { vals[i] = 1; }
        for i in 10..20 { vals[i] = -1; }
        // rest stay 0
        let warp = Warp::new(0, vals);
        let ballot = warp.ballot();
        assert_eq!(ballot.accept, 10);
        assert_eq!(ballot.reject, 10);
        assert_eq!(ballot.neutral, 12);
    }

    #[test]
    fn test_majority_vote() {
        let mut vals = [0i8; WARP_SIZE];
        for i in 0..20 { vals[i] = 1; }
        for i in 20..32 { vals[i] = -1; }
        let warp = Warp::new(0, vals);
        assert_eq!(warp.majority(), 1);
    }

    #[test]
    fn test_warp_all() {
        let warp = Warp::uniform(0, 1);
        assert!(warp.all_nonzero());
        let mut vals = [1i8; WARP_SIZE];
        vals[15] = 0;
        let warp2 = Warp::new(1, vals);
        assert!(!warp2.all_nonzero());
    }

    #[test]
    fn test_block_consensus() {
        let warps = vec![Warp::uniform(0, 1); 8];
        let block = WarpBlock::new(warps);
        let consensus = block.consensus();
        assert_eq!(consensus.decision, 1);
        assert_eq!(consensus.total_threads, 256);
    }

    #[test]
    fn test_hierarchical_reduce() {
        let warps = vec![
            Warp::uniform(0, 1),
            Warp::uniform(1, -1),
        ];
        let block = WarpBlock::new(warps);
        let sum = block.hierarchical_reduce();
        assert_eq!(sum, 0); // balanced
    }

    #[test]
    fn test_any_positive() {
        let warp = Warp::uniform(0, -1);
        assert!(!warp.any_positive());
        let mut vals = [-1i8; WARP_SIZE];
        vals[0] = 1;
        let warp2 = Warp::new(1, vals);
        assert!(warp2.any_positive());
    }
}
