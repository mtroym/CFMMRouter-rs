// cfmm_router_rs/src/lib.rs

// Declare modules
pub mod cfmm;
pub mod objective;
pub mod router;
pub mod types;
pub mod solvers; // For optimization algorithms

// Re-export key types, traits, and functions for easier use by library consumers.
pub use cfmm::CFMM;
pub use objective::{ObjectiveFunction, ObjectiveValue, Gradient};
pub use router::Router;
pub use types::{
    Token, Amount, Price, Fee, // Basic types
    TokenBasket, Reserves, Delta, Lambda, NetTradeVec, DualVariables, // Collections
    ArbitrageResult, RouterResult, // Result structs
    CfmrError, Result, // Error handling
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
} 