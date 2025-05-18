// cfmm_router_rs/src/types.rs
use std::collections::HashMap;
use std::fmt;

// Using String for token identifiers. In a real scenario, this might be an enum,
// a struct with more metadata, or a generic type.
pub type Token = String;

// Using f64 for amounts and prices.
// For financial applications, a dedicated decimal type (e.g., from the `rust_decimal` crate)
// would be more appropriate to avoid floating-point inaccuracies.
pub type Amount = f64;
pub type Price = f64;
pub type Fee = f64; // Represents fee, typically 0.0 < fee <= 1.0

/// Represents a basket of tokens and their corresponding amounts.
/// Positive amounts usually mean tokens received, negative amounts mean tokens tendered.
pub type TokenBasket = HashMap<Token, Amount>;

/// Represents the reserves of a CFMM for various tokens.
pub type Reserves = HashMap<Token, Amount>;

/// Represents amounts tendered to a CFMM.
pub type Delta = HashMap<Token, Amount>;

/// Represents amounts received from a CFMM.
pub type Lambda = HashMap<Token, Amount>;

/// Represents the network trade vector (Ψ in the paper).
/// This is the net result of all trades.
pub type NetTradeVec = HashMap<Token, Amount>;

/// Represents the dual variables (ν in the paper) associated with each token.
pub type DualVariables = HashMap<Token, Price>;

/// Contains the results of an arbitrage operation on a single CFMM.
#[derive(Debug, Clone, PartialEq)]
pub struct ArbitrageResult {
    /// Net profit from the arbitrage, in terms of the numeraire/prices.
    pub profit: Amount,
    /// Tokens tendered to the CFMM.
    pub delta: Delta,
    /// Tokens received from the CFMM.
    pub lambda: Lambda,
}

impl ArbitrageResult {
    pub fn new(profit: Amount, delta: Delta, lambda: Lambda) -> Self {
        Self { profit, delta, lambda }
    }
}

/// Represents the overall result from the router.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterResult {
    /// The maximized utility or objective value.
    pub objective_value: f64,
    /// The net trade vector (Ψ) across all tokens.
    pub net_trades: NetTradeVec,
    /// Individual tendered amounts (Δi) for each CFMM.
    /// The outer Vec corresponds to each CFMM in the order they were provided.
    pub deltas: Vec<Delta>,
    /// Individual received amounts (Λi) for each CFMM.
    /// The outer Vec corresponds to each CFMM in the order they were provided.
    pub lambdas: Vec<Lambda>,
    /// Optimal dual variables (ν).
    pub nu: DualVariables,
}

// Placeholder for specific error types later
#[derive(Debug)]
pub enum CfmrError {
    OptimizationError(String),
    InvalidInput(String),
    CalculationError(String),
    UnimplementedCfmm(String),
}

impl fmt::Display for CfmrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CfmrError::OptimizationError(s) => write!(f, "Optimization Error: {}", s),
            CfmrError::InvalidInput(s) => write!(f, "Invalid Input: {}", s),
            CfmrError::CalculationError(s) => write!(f, "Calculation Error: {}", s),
            CfmrError::UnimplementedCfmm(s) => write!(f, "Unimplemented CFMM: {}", s),
        }
    }
}

impl std::error::Error for CfmrError {}

pub type Result<T> = std::result::Result<T, CfmrError>;

// Helper function to get a value from a HashMap or return a default (0.0 for amounts/prices).
// This is useful because CFMMs might only list tokens they trade, but calculations might involve
// a global set of tokens.
pub(crate) fn get_or_zero(map: &HashMap<Token, f64>, token: &Token) -> f64 {
    *map.get(token).unwrap_or(&0.0)
}

// Helper to sum A_i * (Lambda_i - Delta_i) for all CFMMs
// where A_i is implicitly handled by ensuring Lambda_i and Delta_i use global token names.
pub(crate) fn sum_all_net_flows(
    all_lambdas: &[Lambda],
    all_deltas: &[Delta],
    all_tokens: &[Token],
) -> NetTradeVec {
    let mut total_net_flows: NetTradeVec = HashMap::new();
    for token in all_tokens {
        let mut net_flow_for_token = 0.0;
        for (lambda_i, delta_i) in all_lambdas.iter().zip(all_deltas.iter()) {
            net_flow_for_token += get_or_zero(lambda_i, token) - get_or_zero(delta_i, token);
        }
        if net_flow_for_token.abs() > 1e-9 { // Only include tokens with non-zero net flow, use abs for float comparison
            total_net_flows.insert(token.clone(), net_flow_for_token);
        }
    }
    total_net_flows
} 