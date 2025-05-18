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

        let tokens: Vec<Token> = initial_reserves.keys().cloned().collect();
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
    use crate::types::{Token, Amount, DualVariables, CfmrError};
    use std::collections::HashMap;

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
        let cfmm = setup_cfmm(); // ETH: 1000, USDC: 2_000_000, Fee: 0.3%, Gamma: 0.997. k = 2e9
                                 // Implied price ETH/USDC = 2000
        let mut nu: DualVariables = HashMap::new();
        // Market prices: ETH = 2050 USDC, USDC = 1 (ETH is overvalued in market vs pool)
        // So, we should be able to sell ETH to the pool (Δ_eth > 0) and get USDC (Λ_usdc > 0)
        nu.insert("ETH".to_string(), 2050.0);
        nu.insert("USDC".to_string(), 1.0);

        // Formula for Δ_x_opt from paper (if x is ETH, y is USDC):
        // Δ_x = (sqrt(ν_y * k * γ / ν_x) - R_x) / γ
        // ν_x = 2050 (ETH price in USDC)
        // ν_y = 1 (USDC price in USDC)
        // k = 1000 * 2000000 = 2_000_000_000
        // γ = 0.997
        // R_x = 1000 (ETH reserves)
        // R_y = 2000000 (USDC reserves)

        // sqrt_val = sqrt(1.0 * 2e9 * 0.997 / 2050) = sqrt(1994e6 / 2050) = sqrt(972682.9268) ~= 986.2468
        // delta_eth_opt = (986.2468 - 1000) / 0.997 = -13.7532 / 0.997 ~= -13.79
        // Since delta_eth_opt is negative, this direction (sell ETH) is not profitable with this formula.
        // This suggests the formula is for finding the point where the *pool's* new price matches nu_x/nu_y after trade.
        // If current pool price < nu_x/nu_y, sell Y for X. If current pool price > nu_x/nu_y, sell X for Y.
        // Current pool price for ETH = Ry/Rx = 2000. Market price for ETH = nu_ETH/nu_USDC = 2050.
        // Market thinks ETH is more valuable than pool. So, buy ETH from pool (sell USDC to pool).
        // This means we want to find Δ_y (USDC) and Λ_x (ETH).

        // Let's re-check the logic or use the provided example values if possible.
        // The formulas from the Uniswap v2 paper (Angeris, Kao, Chiang, Noyes, Chitra 2019) might be more direct for optimal amount.
        // Optimal trade amount to make pool price equal to market price nu_x/nu_y:
        // If selling token X (ETH) for token Y (USDC), market price nu_x/nu_y.
        // Amount of X to sell: dx = (sqrt(k * nu_x / nu_y * gamma) - Rx) / gamma
        // Amount of Y to sell: dy = (sqrt(k * nu_y / nu_x * gamma) - Ry) / gamma

        // Here nu_x/nu_y = 2050. Pool price is 2000.
        // We want to sell ETH (token X) if pool_price < market_price. That is not the case here (2000 < 2050 is false, pool thinks ETH is cheaper).
        // We want to sell USDC (token Y) if pool_price_inverted > market_price_inverted (i.e. Rx/Ry > nu_y/nu_x)
        // Rx/Ry = 1/2000 = 0.0005. nu_y/nu_x = 1/2050 = 0.000487.  0.0005 > 0.000487 is true. So sell USDC.

        // Use formulas for selling Y (USDC) for X (ETH):
        // delta_usdc_opt = (sqrt(k * (nu_usdc/nu_eth) * gamma) - Ry) / gamma -- this is wrong, should be nu_x/nu_y for price of X
        // delta_usdc_opt = (sqrt(k * (nu_eth/nu_usdc) / gamma) - Ry) * gamma ; This is from uniswap paper, not routing paper.
        // Let's use the derivation from the solve_arbitrage_subproblem directly:
        // Case 1: Sell X (ETH) for Y (USDC)
        // nu_x = 2050, nu_y = 1, rx=1000, ry=2e6, k=2e9, gamma=0.997
        // val_sqrt_1 = sqrt(nu_y * k * gamma / nu_x) = sqrt(1.0 * 2e9 * 0.997 / 2050) = sqrt(972682.9268) = 986.24689
        // delta_x_opt = (986.24689 - 1000) / 0.997 = -13.7531 / 0.997 = -13.79448... -> not this path

        // Case 2: Sell Y (USDC) for X (ETH)
        // val_sqrt_2 = sqrt(nu_x * k * gamma / nu_y) = sqrt(2050 * 2e9 * 0.997 / 1.0) = sqrt(4.08789e12) = 2021853.09
        // delta_y_opt = (val_sqrt_2 - ry) / gamma = (2021853.09 - 2000000) / 0.997 = 21853.09 / 0.997 = 21918.8465

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        // Expected: Sell USDC (token Y) to get ETH (token X)
        // delta_y_opt (USDC sent) approx 21918.85
        // ry_new = ry + gamma * delta_y_opt = 2e6 + 0.997 * 21918.85 = 2e6 + 21853.09 = 2021853.09
        // rx_new = k / ry_new = 2e9 / 2021853.09 = 989.191
        // lambda_x_opt (ETH received) = rx - rx_new = 1000 - 989.191 = 10.809
        // Profit = nu_x * lambda_x_opt - nu_y * delta_y_opt
        //        = 2050 * 10.809 - 1.0 * 21918.85
        //        = 22158.45 - 21918.85 = 239.60

        assert!(result.profit > 0.0, "Expected positive profit");
        let delta_usdc = result.delta.get("USDC").unwrap();
        let lambda_eth = result.lambda.get("ETH").unwrap();

        assert!(*delta_usdc > 21918.0 && *delta_usdc < 21919.0); // Around 21918.85
        assert!(*lambda_eth > 10.80 && *lambda_eth < 10.81);    // Around 10.809
        assert!(result.profit > 239.5 && result.profit < 239.7); // Around 239.60
    }

    #[test]
    fn solve_arb_sell_usdc_for_eth() {
        let cfmm = setup_cfmm(); // ETH: 1000, USDC: 2_000_000, Fee: 0.3%, Gamma: 0.997. k = 2e9
                                 // Implied price ETH/USDC = 2000
        let mut nu: DualVariables = HashMap::new();
        // Market prices: ETH = 1950 USDC, USDC = 1 (ETH is undervalued in market vs pool)
        // So, we should be able to sell ETH to market, meaning buy ETH from pool and sell to market.
        // To buy ETH from pool, we provide USDC. So Δ_usdc > 0, Λ_eth > 0.
        // OR, equivalently, pool's ETH is overpriced relative to market. So, sell ETH *to* the pool.
        // Market thinks ETH is less valuable (1950) than pool (2000). So sell ETH to pool.
        // This means Δ_eth > 0, Λ_usdc > 0.

        nu.insert("ETH".to_string(), 1950.0);
        nu.insert("USDC".to_string(), 1.0);

        // Case 1: Sell X (ETH) for Y (USDC)
        // nu_x = 1950, nu_y = 1, rx=1000, ry=2e6, k=2e9, gamma=0.997
        // val_sqrt_1 = sqrt(nu_y * k * gamma / nu_x) = sqrt(1.0 * 2e9 * 0.997 / 1950) = sqrt(1022564.1025) = 1011.2191
        // delta_x_opt = (1011.2191 - 1000) / 0.997 = 11.2191 / 0.997 = 11.2528

        let result = cfmm.solve_arbitrage_subproblem(&nu).unwrap();

        // Expected: Sell ETH (token X) to get USDC (token Y)
        // delta_x_opt (ETH sent) approx 11.2528
        // rx_new = rx + gamma * delta_x_opt = 1000 + 0.997 * 11.2528 = 1000 + 11.2191 = 1011.2191
        // ry_new = k / rx_new = 2e9 / 1011.2191 = 1977815.0
        // lambda_y_opt (USDC received) = ry - ry_new = 2000000 - 1977815.0 = 22185.0
        // Profit = nu_y * lambda_y_opt - nu_x * delta_x_opt
        //        = 1.0 * 22185.0 - 1950 * 11.2528
        //        = 22185.0 - 21942.96 = 242.04

        assert!(result.profit > 0.0, "Expected positive profit");
        let delta_eth = result.delta.get("ETH").unwrap();
        let lambda_usdc = result.lambda.get("USDC").unwrap();

        assert!(*delta_eth > 11.25 && *delta_eth < 11.26);       // Around 11.2528
        assert!(*lambda_usdc > 22184.0 && *lambda_usdc < 22186.0); // Around 22185.0
        assert!(result.profit > 242.0 && result.profit < 242.1); // Around 242.04
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
} 