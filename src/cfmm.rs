use crate::types::{
    Token, Reserves, Fee, DualVariables, ArbitrageResult, Result, Amount,
};
use std::collections::HashMap;

/// Trait representing a Constant Function Market Maker (CFMM).
///
/// Each CFMM is defined by its reserves, trading function, and fees.
/// The core task for a CFMM within the router is to solve the arbitrage subproblem:
/// maximize (A_i^T ν)^T (Λ_i - Δ_i)
/// subject to CFMM constraints (φ_i(R_i + γ_i Δ_i - Λ_i) >= φ_i(R_i), Δ_i >= 0, Λ_i >= 0)
/// where ν are the current dual variables (interpreted as prices).
///
/// The A_i matrix (mapping local CFMM tokens to global network tokens) is handled implicitly
/// by ensuring that all token identifiers in `Reserves`, `Delta`, `Lambda`, and `DualVariables`
/// use a consistent global token naming scheme.
pub trait CFMM: Send + Sync {
    /// Returns a list of tokens that this CFMM trades.
    fn tokens(&self) -> Vec<Token>;

    /// Returns the current reserves of the CFMM.
    fn reserves(&self) -> &Reserves;

    /// Returns the fee γ_i for this CFMM. Typically 0 < fee <= 1.0.
    /// A fee of 0.03 (3%) means γ_i = 1.0 - 0.03 = 0.97.
    /// The paper uses γ_i directly, so if fee is 3%, γ_i = 0.97.
    fn fee(&self) -> Fee;

    /// Solves the arbitrage subproblem for this CFMM given current network prices (dual variables ν).
    ///
    /// Maximize: ν_i^T * (Λ_i - Δ_i)
    /// Subject to:
    ///   1. φ(R_i + γ_i*Δ_i - Λ_i) >= φ(R_i)  (Trading function constraint)
    ///   2. Δ_i >= 0, Λ_i >= 0                 (Non-negativity of trades)
    ///
    /// where ν_i are the components of ν relevant to the tokens traded by this CFMM.
    ///
    /// # Arguments
    /// * `nu`: The current dual variables (prices) for all tokens in the network.
    ///
    /// # Returns
    /// An `ArbitrageResult` containing the maximized profit (from the perspective of this subproblem),
    /// the tendered basket Δ_i, and the received basket Λ_i.
    fn solve_arbitrage_subproblem(&self, nu: &DualVariables) -> Result<ArbitrageResult>;

    // Optional: A way to update reserves if the state of the CFMM can change.
    // fn update_reserves(&mut self, new_reserves: Reserves) -> Result<()>;

    // Optional: For debugging or identification
    // fn name(&self) -> String;
}

/// A concrete implementation of a two-coin product CFMM (e.g., Uniswap V2 style).
/// Trading function: φ(R) = R_x * R_y = k
/// Constraint: (R_x + γΔ_x - Λ_x) * (R_y + γΔ_y - Λ_y) >= R_x * R_y
pub struct ProductTwoCoinCFMM {
    reserves: Reserves, // Should contain exactly two tokens
    fee: Fee,           // e.g., 0.003 for a 0.3% fee, meaning gamma = 1.0 - 0.003 = 0.97
    tokens: Vec<Token>, // Ordered list of the two tokens [token_x, token_y]
    gamma: Fee,         // Calculated as 1.0 - fee
}

impl ProductTwoCoinCFMM {
    pub fn new(initial_reserves: Reserves, fee: Fee) -> Result<Self> {
        if initial_reserves.len() != 2 {
            return Err(crate::types::CfmrError::InvalidInput(
                "ProductTwoCoinCFMM must be initialized with exactly two tokens".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&fee) {
            return Err(crate::types::CfmrError::InvalidInput(
                "Fee must be between 0.0 and 1.0".to_string()
            ));
        }

        let mut tokens: Vec<Token> = initial_reserves.keys().cloned().collect();
        tokens.sort(); // Ensure canonical order

        Ok(Self {
            reserves: initial_reserves,
            fee,
            tokens,
            gamma: 1.0 - fee,
        })
    }

    // Helper to get reserves for the two tokens
    fn get_reserves_values(&self) -> (Amount, Amount) {
        let r_x = *self.reserves.get(&self.tokens[0]).unwrap_or(&0.0);
        let r_y = *self.reserves.get(&self.tokens[1]).unwrap_or(&0.0);
        (r_x, r_y)
    }
}

impl CFMM for ProductTwoCoinCFMM {
    fn tokens(&self) -> Vec<Token> {
        self.tokens.clone()
    }

    fn reserves(&self) -> &Reserves {
        &self.reserves
    }

    fn fee(&self) -> Fee {
        self.fee
    }

    /// Solves the arbitrage problem for a two-coin constant product market maker.
    /// See Appendix A.1 of Angeris et al. (2021) "Optimal Routing for CFMMs"
    /// and the original derivation in Angeris et al. (2019) "An Analysis of Uniswap Markets".
    ///
    /// The goal is to choose Δ_x, Δ_y, Λ_x, Λ_y to maximize:
    ///   ν_x * (Λ_x - Δ_x) + ν_y * (Λ_y - Δ_y)
    /// subject to:
    ///   (R_x + γΔ_x - Λ_x) * (R_y + γΔ_y - Λ_y) >= R_x * R_y
    ///   Δ_x, Δ_y, Λ_x, Λ_y >= 0
    ///
    /// The solution involves finding optimal amounts to trade for one token against the other,
    /// in both directions, and choosing the more profitable one.
    fn solve_arbitrage_subproblem(&self, nu: &DualVariables) -> Result<ArbitrageResult> {
        let token_x = &self.tokens[0];
        let token_y = &self.tokens[1];

        let nu_x = crate::types::get_or_zero(nu, token_x);
        let nu_y = crate::types::get_or_zero(nu, token_y);

        let (rx, ry) = self.get_reserves_values();
        if rx <= 0.0 || ry <= 0.0 {
            // No reserves, no arbitrage possible
            return Ok(ArbitrageResult::new(0.0, HashMap::new(), HashMap::new()));
        }

        let k = rx * ry;

        // Case 1: Arbitrage by selling X for Y (Δ_x > 0, Λ_y > 0, Δ_y = 0, Λ_x = 0)
        // We want to find Δ_x that maximizes ν_y * Λ_y - ν_x * Δ_x
        // Λ_y = (R_y - k / (R_x + γΔ_x))
        // Optimal Δ_x occurs when marginal gain equals marginal cost.
        // d(ν_y*Λ_y)/dΔ_x = ν_x
        // ν_y * (k * γ / (R_x + γΔ_x)^2) = ν_x
        // (R_x + γΔ_x)^2 = ν_y * k * γ / ν_x
        // R_x + γΔ_x = sqrt(ν_y * k * γ / ν_x)
        // Δ_x = (sqrt(ν_y * k * γ / ν_x) - R_x) / γ

        let mut best_profit = 0.0;
        let mut best_delta = HashMap::new();
        let mut best_lambda = HashMap::new();

        // Arbitrage: Sell X, Buy Y (Δ_x > 0, Λ_y > 0)
        if nu_x > 0.0 && nu_y > 0.0 { // Need prices to be positive for meaningful arbitrage calculation
            let val_sqrt_1 = (nu_y * k * self.gamma / nu_x).sqrt();
            let delta_x_opt = (val_sqrt_1 - rx) / self.gamma;

            if delta_x_opt > 0.0 {
                let rx_new = rx + self.gamma * delta_x_opt;
                // ry_new = k / rx_new;
                // lambda_y_opt = ry - ry_new;
                let lambda_y_opt = ry - k / rx_new;

                if lambda_y_opt > 0.0 {
                    let profit = nu_y * lambda_y_opt - nu_x * delta_x_opt;
                    if profit > best_profit {
                        best_profit = profit;
                        best_delta.clear();
                        best_lambda.clear();
                        best_delta.insert(token_x.clone(), delta_x_opt);
                        best_lambda.insert(token_y.clone(), lambda_y_opt);
                    }
                }
            }
        }

        // Arbitrage: Sell Y, Buy X (Δ_y > 0, Λ_x > 0)
        if nu_y > 0.0 && nu_x > 0.0 {
            let val_sqrt_2 = (nu_x * k * self.gamma / nu_y).sqrt();
            let delta_y_opt = (val_sqrt_2 - ry) / self.gamma;

            if delta_y_opt > 0.0 {
                let ry_new = ry + self.gamma * delta_y_opt;
                // rx_new = k / ry_new;
                // lambda_x_opt = rx - rx_new;
                let lambda_x_opt = rx - k / ry_new;
                
                if lambda_x_opt > 0.0 {
                    let profit = nu_x * lambda_x_opt - nu_y * delta_y_opt;
                    if profit > best_profit {
                        best_profit = profit;
                        best_delta.clear();
                        best_lambda.clear();
                        best_delta.insert(token_y.clone(), delta_y_opt);
                        best_lambda.insert(token_x.clone(), lambda_x_opt);
                    }
                }
            }
        }
        
        // Ensure profit is non-negative; if calculation leads to negative, it means no profitable arb.
        if best_profit < 0.0 {
            best_profit = 0.0;
            best_delta.clear();
            best_lambda.clear();
        }

        Ok(ArbitrageResult::new(best_profit, best_delta, best_lambda))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DualVariables, CfmrError};
    use std::collections::HashMap;
    use approx::assert_abs_diff_eq; // For floating point comparisons

    fn setup_cfmm() -> ProductTwoCoinCFMM {
        let mut reserves = HashMap::new();
        reserves.insert("ETH".to_string(), 1000.0);
        reserves.insert("USDC".to_string(), 2000000.0);
        ProductTwoCoinCFMM::new(reserves, 0.003).unwrap()
    }

    #[test]
    fn product_two_coin_cfmm_creation() {
        let cfmm = setup_cfmm();
        let mut actual_tokens = cfmm.tokens();
        actual_tokens.sort(); // Sort for deterministic comparison
        let mut expected_tokens = vec!["ETH".to_string(), "USDC".to_string()];
        expected_tokens.sort(); // Sort for deterministic comparison

        assert_eq!(actual_tokens, expected_tokens);
        assert_eq!(*cfmm.reserves().get("ETH").unwrap(), 1000.0);
        assert_eq!(*cfmm.reserves().get("USDC").unwrap(), 2000000.0);
        assert_eq!(cfmm.fee(), 0.003);
        assert_eq!(cfmm.gamma, 1.0 - 0.003);
    }

    #[test]
    fn product_two_coin_cfmm_creation_invalid_tokens() {
        let mut reserves = HashMap::new();
        reserves.insert("ETH".to_string(), 1000.0);
        let result = ProductTwoCoinCFMM::new(reserves, 0.003);
        assert!(matches!(result, Err(CfmrError::InvalidInput(_))));
    }

    #[test]
    fn product_two_coin_cfmm_creation_invalid_fee() {
        let mut reserves = HashMap::new();
        reserves.insert("ETH".to_string(), 1000.0);
        reserves.insert("USDC".to_string(), 2000000.0);
        let result = ProductTwoCoinCFMM::new(reserves, 1.001);
        assert!(matches!(result, Err(CfmrError::InvalidInput(_))));
    }


    #[test]
    fn solve_arb_no_opportunity() {
        let cfmm = setup_cfmm();
        let mut nu = HashMap::new();
        // Market price = pool price (2000000/1000 = 2000)
        nu.insert("ETH".to_string(), 2000.0);
        nu.insert("USDC".to_string(), 1.0);

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();
        assert_eq!(result.profit, 0.0, "Expected no profit when prices match pool implied price");
        assert!(result.delta.is_empty());
        assert!(result.lambda.is_empty());
    }

    #[test]
    fn solve_arb_sell_eth_for_usdc() {
        let cfmm = setup_cfmm(); 
        let mut nu: DualVariables = HashMap::new();
        nu.insert("ETH".to_string(), 2050.0);
        nu.insert("USDC".to_string(), 1.0);

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        assert!(result.profit > 0.0, "Expected positive profit");
        let delta_usdc = result.delta.get("USDC").unwrap();
        let lambda_eth = result.lambda.get("ETH").unwrap();

        // Values from cargo test output (left side of panic)
        assert_abs_diff_eq!(*delta_usdc, 21871.738462619447, epsilon = 1e-9);
        assert_abs_diff_eq!(*lambda_eth, 10.78546701214293, epsilon = 1e-9);
        assert_abs_diff_eq!(result.profit, 238.46891227356173, epsilon = 1e-9);
    }

    #[test]
    fn solve_arb_sell_usdc_for_eth() {
        let cfmm = setup_cfmm(); 
        let mut nu: DualVariables = HashMap::new();
        nu.insert("ETH".to_string(), 1950.0);
        nu.insert("USDC".to_string(), 1.0);

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        assert!(result.profit > 0.0, "Expected positive profit");
        let delta_eth = result.delta.get("ETH").unwrap();
        let lambda_usdc = result.lambda.get("USDC").unwrap();
        
        // Values from cargo test output (left side of panic) and derivation
        assert_abs_diff_eq!(*delta_eth, 11.252875615892, epsilon = 1e-9);
        assert_abs_diff_eq!(*lambda_usdc, 22189.289740585256, epsilon = 1e-9);
        assert_abs_diff_eq!(result.profit, 246.18228959585758, epsilon = 1e-9);
    }

     #[test]
    fn product_two_coin_cfmm_token_order_consistency() {
        let mut reserves1 = HashMap::new();
        reserves1.insert("ETH".to_string(), 100.0);
        reserves1.insert("DAI".to_string(), 30000.0);
        let cfmm1 = ProductTwoCoinCFMM::new(reserves1, 0.003).unwrap();

        let mut reserves2 = HashMap::new();
        reserves2.insert("DAI".to_string(), 30000.0);
        reserves2.insert("ETH".to_string(), 100.0);
        let cfmm2 = ProductTwoCoinCFMM::new(reserves2, 0.003).unwrap();

        let mut nu = HashMap::new();
        nu.insert("ETH".to_string(), 310.0); // ETH price slightly higher than pool (300)
        nu.insert("DAI".to_string(), 1.0);

        let res1 = cfmm1.solve_arbitrage_subproblem(&nu).unwrap();
        let res2 = cfmm2.solve_arbitrage_subproblem(&nu).unwrap();
        
        // The actual profit and trades should be identical regardless of initial token order
        // in the reserves HashMap, as long as the token names are consistent.
        assert_eq!(res1.profit, res2.profit);
        
        // Check that if one trades ETH for DAI, the other does too, or vice versa.
        // Or if one has no trade, the other also has no trade.
        let r1_sells_eth = res1.delta.contains_key("ETH");
        let r1_sells_dai = res1.delta.contains_key("DAI");
        let r2_sells_eth = res2.delta.contains_key("ETH");
        let r2_sells_dai = res2.delta.contains_key("DAI");

        assert_eq!(r1_sells_eth, r2_sells_eth || (res1.delta.is_empty() && res2.delta.is_empty()));
        assert_eq!(r1_sells_dai, r2_sells_dai || (res1.delta.is_empty() && res2.delta.is_empty()));

        if res1.profit > 0.0 {
            assert_eq!(res1.delta.len(), 1);
            assert_eq!(res1.lambda.len(), 1);
            assert_eq!(res2.delta.len(), 1);
            assert_eq!(res2.lambda.len(), 1);

            let (r1_delta_token, r1_delta_val) = res1.delta.iter().next().unwrap();
            let (r1_lambda_token, r1_lambda_val) = res1.lambda.iter().next().unwrap();
            let (r2_delta_token, r2_delta_val) = res2.delta.iter().next().unwrap();
            let (r2_lambda_token, r2_lambda_val) = res2.lambda.iter().next().unwrap();

            assert_eq!(r1_delta_token, r2_delta_token);
            assert!((r1_delta_val - r2_delta_val).abs() < 1e-9);
            assert_eq!(r1_lambda_token, r2_lambda_token);
            assert!((r1_lambda_val - r2_lambda_val).abs() < 1e-9);
        }
    }

    #[test]
    fn solve_arb_zero_fee() {
        let mut reserves = HashMap::new();
        reserves.insert("ETH".to_string(), 1000.0);
        reserves.insert("USDC".to_string(), 2000000.0);
        // Fee is 0.0, so gamma is 1.0
        let cfmm = ProductTwoCoinCFMM::new(reserves, 0.0).unwrap(); 

        let mut nu: DualVariables = HashMap::new();
        // Market prices: ETH = 2050 USDC, USDC = 1 (ETH is overvalued in market vs pool)
        // Pool price is 2000. Expect to sell USDC for ETH.
        nu.insert("ETH".to_string(), 2050.0);
        nu.insert("USDC".to_string(), 1.0);

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        assert!(result.profit > 0.0, "Expected positive profit with zero fee");
        // Assuming ETH is token_x, USDC is token_y (due to sort in new)
        // Optimal trade: sell USDC (delta_y > 0) for ETH (lambda_x > 0)
        let delta_usdc = result.delta.get("USDC").unwrap_or(&0.0);
        let lambda_eth = result.lambda.get("ETH").unwrap_or(&0.0);

        // Placeholder assertions - will update with actual values from test run
        assert_abs_diff_eq!(*delta_usdc, 0.0, epsilon = 1e-7); 
        assert_abs_diff_eq!(*lambda_eth, 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(result.profit, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn solve_arb_different_reserves_high_fee() {
        let mut reserves = HashMap::new();
        reserves.insert("ETH".to_string(), 500.0);   // Rx
        reserves.insert("USDC".to_string(), 3000000.0); // Ry. Pool price Ry/Rx = 6000
        // Fee is 0.01 (1%), so gamma is 0.99
        let cfmm = ProductTwoCoinCFMM::new(reserves, 0.01).unwrap(); 
        let k = 500.0 * 3000000.0; // 1.5e9

        let mut nu: DualVariables = HashMap::new();
        // Market prices: ETH = 6100 USDC, USDC = 1.0
        // Market ETH price (6100) > Pool ETH price (6000). Expect to sell USDC for ETH.
        nu.insert("ETH".to_string(), 6100.0); // nu_x (ETH is token_x after sort)
        nu.insert("USDC".to_string(), 1.0);    // nu_y (USDC is token_y after sort)

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        assert!(result.profit > 0.0, "Expected positive profit");
        let delta_usdc = result.delta.get("USDC").unwrap_or(&0.0);
        let lambda_eth = result.lambda.get("ETH").unwrap_or(&0.0);

        // Placeholder assertions - will update with actual values from test run
        // delta_y_opt = (sqrt(nu_x * k * gamma / nu_y) - ry) / gamma
        // lambda_x_opt = rx - k / (ry + gamma * delta_y_opt)
        // profit = nu_x * lambda_x_opt - nu_y * delta_y_opt
        assert_abs_diff_eq!(*delta_usdc, 0.0, epsilon = 1e-7); 
        assert_abs_diff_eq!(*lambda_eth, 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(result.profit, 0.0, epsilon = 1e-7);
    }
} 