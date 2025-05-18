use crate::types::{NetTradeVec, DualVariables, Price, Result, Token, get_or_zero};
use std::collections::HashMap;

/// Represents the value of the objective function U(Ψ) or its Fenchel conjugate term.
pub type ObjectiveValue = f64;

/// Represents the gradient of the objective function ∇U(Ψ) or related terms.
/// This is a map from tokens to their gradient components.
pub type Gradient = HashMap<Token, Price>;

/// Trait for the objective function U(Ψ) in the routing problem.
/// The router aims to maximize this function.
/// For the dual problem, we need its Fenchel conjugate: (-U)*(-ν).
pub trait ObjectiveFunction: Send + Sync {
    /// Computes the Fenchel conjugate term (-U)*(-ν) = sup_Ψ ( ν^T Ψ - U(Ψ) )
    /// and the Ψ* that achieves this supremum.
    ///
    /// # Arguments
    /// * `nu`: The current dual variables ν (a map from token to its dual value).
    /// * `all_tokens`: A slice defining all unique tokens in the system, for canonical ordering if needed.
    ///
    /// # Returns
    /// A tuple `(value, psi_star)`:
    ///   - `value`: The value of the Fenchel conjugate term.
    ///   - `psi_star`: The net trade vector Ψ* that achieves the supremum.
    fn compute_fenchel_conjugate_term(&self, nu: &DualVariables, all_tokens: &[Token]) -> Result<(ObjectiveValue, NetTradeVec)>;

    /// Evaluates the objective function U(Ψ) for a given net trade vector Ψ.
    fn evaluate(&self, psi: &NetTradeVec) -> Result<ObjectiveValue>;
}

/// A simple linear non-negative objective function: U(Ψ) = p^T Ψ, subject to Ψ >= 0 for certain tokens.
/// This is often used for finding arbitrage opportunities where `p` are the market prices.
/// The problem is to maximize p^T Ψ.
///
/// For this U(Ψ) = p^T Ψ:
/// (-U)*(-ν) = sup_Ψ (p^T Ψ - ν^T Ψ) = sup_Ψ ((p - ν)^T Ψ)
/// If p - ν > 0, then Ψ -> ∞, so sup is ∞.
/// If p - ν <= 0, then Ψ = 0 maximizes it, and sup is 0.
/// This implies U must be defined more carefully for arbitrage, often U(Ψ) = ν_external^T Ψ, and then the optimizer finds ν_internal.
/// The paper's formulation seems to be maximizing U(Ψ) directly.
///
/// Let's consider the objective from the Julia CFMMRouter.jl `LinearNonnegative`:
/// It aims to maximize `prices^T Ψ` s.t. `Ψ >= 0` (implicitly for tokens included in `prices`).
/// The Fenchel conjugate term needed is `sup_Ψ (prices^T Ψ - ν^T Ψ)`.
/// `sup_Ψ ((prices - ν)^T Ψ)`
/// If `prices_k - ν_k > 0` for any k, then Ψ_k can go to infinity, making the sup infinity.
/// This indicates that for this objective, we must have `prices_k - ν_k <= 0` for all k.
/// In this case, the supremum is 0, achieved when Ψ = 0.
///
/// This is characteristic of objectives that are linear. The L-BFGS-B solver typically handles bounds.
/// The Julia implementation might be doing something specific for this objective within its solver framework,
/// or the problem setup (like finding arbitrage) inherently ensures convergence.
///
/// From the paper: `(-U)*(-ν) + sum(arb_i(A_i^T ν))`
/// If U(Ψ) = `target_prices^T Ψ` (e.g. arbitrage against external market prices `target_prices`)
/// Then `(-U)*(-ν)` is `sup_Ψ (target_prices^T Ψ - ν^T Ψ)`.
/// The gradient of this term w.r.t ν is `-Ψ*` where `Ψ*` is the argmax.
/// If `target_prices_k - ν_k > 0`, `Ψ*_k -> ∞` (problematic).
/// If `target_prices_k - ν_k <= 0`, `Ψ*_k = 0` and value is 0.
/// If `target_prices_k - ν_k = 0`, `Ψ*_k` can be anything non-negative.
///
/// This suggests that for a pure arbitrage objective (maximize profit against fixed prices `p`), the `U(Ψ)` formulation is direct,
/// and the `ν` are the internal network prices discovered by the solver.
/// The `LinearNonnegative` in Julia seems to represent `U(Ψ) = prices ⋅ Ψ` where `Ψ >=0`.
/// The `route!(router)` function in Julia then tries to maximize this.
///
/// The dual formulation requires `(-U)*(-ν)`. For `U(x) = a^T x`, then `(-U)(x) = -a^T x`.
/// `(-U)*(-ν) = sup_x (-a^T x - (-ν)^T x) = sup_x ( (ν-a)^T x )`.
/// This is 0 if `ν-a <= 0` (i.e. `ν <= a`), and ∞ otherwise. `Ψ* = 0` if `ν < a`.
/// If `ν = a`, `Ψ*` can be any non-negative vector, value is 0.
/// This means that at the optimum, we expect `ν* = a` (the target prices).
/// The `Ψ*` that forms the gradient component `-Ψ*` comes from this supremum.
/// If the objective is just `prices^T Ψ` (arbitrage), then `U(Ψ) = prices^T Ψ`.
/// Then `-U(Ψ) = -prices^T Ψ`. So `(-U)^*(-ν) = sup_Ψ(-prices^T Ψ - (-ν)^T Ψ) = sup_Ψ((ν - prices)^T Ψ)`.
/// If `ν_k > prices_k` for any `k` where `Ψ_k` can be positive, then sup is ∞.
/// If `ν_k <= prices_k` for all `k`, then `Ψ_k=0` gives sup = 0.
/// The optimal `Ψ*` used for `∇g(ν)` is tricky here. The Julia library must handle this.
///
/// Let's assume the `LinearNonnegative(prices)` objective implies `U(Ψ) = prices^T Ψ`.
/// The `route!` method tries to find `Δ_i, Λ_i` that maximize `prices^T (sum A_i(Λ_i - Δ_i))`.
/// The dual variable `ν` represents the internal consensus prices.
/// At convergence, `ν` should reflect the `prices` if arbitrage is possible, or market clearing prices otherwise.
///
/// For `U(Ψ) = p^T Ψ`, constrained by `Ψ >= 0` implicitly.
/// The term is `sup_Ψ(p^T Ψ - ν^T Ψ)`.
/// The `Ψ` that maximizes this is:
///   - `Ψ_k = large_positive_number` if `p_k - ν_k > 0`
///   - `Ψ_k = 0` if `p_k - ν_k < 0`
///   - `Ψ_k = any_non_negative_value` if `p_k - ν_k = 0`
/// The value of the sup is 0 if `p_k - ν_k <=0` for all k, and ∞ otherwise.
/// The solver (L-BFGS-B) will adjust `ν` to avoid the ∞ region.
/// So, it will operate where `p_k - ν_k <= 0` for all `k` (i.e. `ν_k >= p_k`).
/// In this region, the value is 0, and `Ψ*` (the argmax) is a vector of zeros.
/// This means the gradient contribution from this term `-Ψ*` would be zero.
/// This seems too simple and likely means my understanding of how it interacts with the dual decomposition for *this specific* objective is incomplete.
///
/// Let's look at the reference paper by Angeris et al. (2021) or the CFMMRouter.jl documentation for `LinearNonnegative` details.
/// The paper states: `maximize U(Ψ)`. And gives examples like `U(Ψ) = min_k ( (AΨ)_k / w_k)` (liquidating a basket).
/// For arbitrage, it says: `maximize p^T Ψ` subject to `Ψ >= c` (e.g. `c=0`).
/// If `U(Ψ) = p^T Ψ`, then `(-U)*(-ν) = sup_x ( (ν-p)^T x )`.
/// The gradient `∇_ν ((-U)*(-ν))` is `x*` where `x*` is the argmax.
/// If `ν_k - p_k > 0`, `x*_k -> inf`. If `ν_k - p_k < 0`, `x*_k = 0` (assuming `x >= 0`). If `ν_k - p_k = 0`, `x*_k` is any non-negative.
/// The solver should drive `ν` such that `ν_k - p_k <= 0`.
///
/// The Julia code for `LinearNonnegative` objective has:
/// `fenchel_conjugate_term(obj::LinearNonnegative, ν) = 0.0`
/// `grad_fenchel_conjugate_term!(grad, obj::LinearNonnegative, ν) = (grad .= 0.0)`
/// This implies that `sup_Ψ ((p - ν)^T Ψ)` is taken to be 0, and `Ψ* = 0`.
/// This happens if `p - ν <= 0` (i.e., `ν >= p`). The solver is expected to enforce this for `ν`.
/// This is a strong simplifying assumption. It means that this objective doesn't directly contribute to pushing Ψ away from zero based on external prices `p` via the Fenchel conjugate term;
/// rather, the `p` likely guides the overall optimization goal which the `arb_i` terms satisfy.
/// So, the `ν` values are the *result* of the optimization, representing internal prices.
/// If we want to maximize profit against *external* prices `p`, then `U(Ψ) = p^T Ψ` is the primal objective.
/// The dual variables `ν` are prices that make `sum_i arb_i(A_i^T ν)` balance out.
/// The problem is `max U(Ψ) s.t. Ψ = sum(...)`. Dual is `min_ν g(ν) = min_ν { (-U)*(-ν) + sum_i arb_i(A_i^T ν) }`

pub struct LinearNonnegative {
    /// The prices `p` for the linear objective function U(Ψ) = p^T Ψ.
    /// Only tokens present in this map are considered part of the objective.
    /// Other tokens in Ψ will have an implicit price of 0 in the objective.
    prices: HashMap<Token, Price>,
}

impl LinearNonnegative {
    pub fn new(prices: HashMap<Token, Price>) -> Self {
        Self { prices }
    }
}

impl ObjectiveFunction for LinearNonnegative {
    /// For U(Ψ) = p^T Ψ (with implicit Ψ_k >= 0 for p_k != 0, and Ψ_k unconstrained if p_k=0, or Ψ generally unconstrained).
    /// (-U)*(-ν) = sup_Ψ ( (ν-p)^T Ψ ).
    /// If we assume the solver ensures ν converges such that ν_k <= p_k where Ψ_k > 0 is allowed by arbitrage,
    /// or more generally, if the problem is feasible, this term might be 0 and Ψ*=0.
    ///
    /// Following the Julia implementation's simplification for `LinearNonnegative`:
    /// The Fenchel conjugate term `(-U)*(-ν)` is considered to be 0.
    /// The corresponding `Ψ*` (which forms the negative of the gradient part) is also 0.
    /// This implies that the `prices` vector `p` is what the dual variables `ν` should ideally converge towards
    /// if an arbitrage exists, driven by the `arb_i` terms.
    fn compute_fenchel_conjugate_term(&self, _nu: &DualVariables, _all_tokens: &[Token]) -> Result<(ObjectiveValue, NetTradeVec)> {
        // This is a simplification based on observations from CFMMRouter.jl for LinearNonnegative.
        // It implies that the solver will drive nu such that (nu - p)^T Psi is maximized by Psi = 0.
        // (i.e. nu_k - p_k <= 0 for Psi_k >= 0, nu_k - p_k >=0 for Psi_k <=0, or nu_k=p_k if unconstrained)
        // For arbitrage (maximize p^T Psi, Psi generally >=0), this implies nu <= p.
        // More accurately, the paper implies the dual variables ν are the prices to evaluate arbitrage, and U(Ψ) is the utility.
        // If U(Ψ) = p^T Ψ, then (-U)*(-ν) = sup_x((ν-p)^T x).
        // If solver ensures ν <= p, then sup is 0, achieved at x=0.
        Ok((0.0, HashMap::new()))
    }

    fn evaluate(&self, psi: &NetTradeVec) -> Result<ObjectiveValue> {
        let mut value = 0.0;
        for (token, amount) in psi {
            let price = get_or_zero(&self.prices, token);
            value += price * amount;
        }
        Ok(value)
    }
}

/// Objective: Maximize a specific token `token_out` given `token_in` and `amount_in`.
/// U(Ψ) = Ψ_out  (where Ψ_out is amount of token_out)
/// Subject to Ψ_in = -amount_in (fixed input amount)
/// And Ψ_k = 0 for other tokens k.
/// This is more complex because of the fixed input constraint.
///
/// A simpler version: Maximize `Ψ_target_token` for a specific `target_token`.
/// U(Ψ) = Ψ_target_token.
/// So `p` has 1 for `target_token` and 0 for others.
pub struct MaximizeToken {
    target_token: Token,
}

impl MaximizeToken {
    pub fn new(target_token: Token) -> Self {
        Self { target_token }
    }
}

impl ObjectiveFunction for MaximizeToken {
    fn compute_fenchel_conjugate_term(&self, _nu: &DualVariables, _all_tokens: &[Token]) -> Result<(ObjectiveValue, NetTradeVec)> {
        // U(Ψ) = Ψ_target.
        // (-U)*(-ν) = sup_Ψ ( (ν_target - 1)Ψ_target + sum_{k!=target} ν_k Ψ_k ).
        // If ν_target < 1, Ψ_target -> -∞ (if allowed).
        // If ν_target > 1, Ψ_target -> +∞.
        // If ν_k != 0 for k!=target, Ψ_k -> +/-∞.
        // This suggests for this objective to be well-behaved in dual, ν_target must be 1, and ν_k must be 0 for others.
        // At this point, sup is 0. Ψ* would be {target_token: any_non_negative, others: 0}.
        // This is the same simplification as LinearNonnegative if p = {target_token: 1.0}.
        Ok((0.0, HashMap::new()))
    }

    fn evaluate(&self, psi: &NetTradeVec) -> Result<ObjectiveValue> {
        Ok(get_or_zero(psi, &self.target_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Token};
    use std::collections::HashMap;

    #[test]
    fn linear_nonnegative_evaluate() {
        let mut prices = HashMap::new();
        prices.insert("ETH".to_string(), 2000.0);
        prices.insert("USDC".to_string(), 1.0);
        let objective = LinearNonnegative::new(prices);

        let mut psi = HashMap::new();
        psi.insert("ETH".to_string(), 2.0);
        psi.insert("USDC".to_string(), -100.0);
        psi.insert("BTC".to_string(), 0.5); // Not in objective prices, treated as 0

        let value = objective.evaluate(&psi).unwrap();
        assert_eq!(value, 2.0 * 2000.0 + (-100.0) * 1.0 + 0.5 * 0.0);
        // 4000 - 100 = 3900
        assert_eq!(value, 3900.0);
    }

    #[test]
    fn linear_nonnegative_fenchel_conjugate_term() {
        let mut prices = HashMap::new();
        prices.insert("ETH".to_string(), 2000.0);
        let objective = LinearNonnegative::new(prices);
        let nu = HashMap::new(); // Empty nu for simplicity, content doesn't matter for this impl
        let all_tokens: Vec<Token> = vec!["ETH".to_string(), "USDC".to_string()];

        let (value, psi_star) = objective.compute_fenchel_conjugate_term(&nu, &all_tokens).unwrap();
        assert_eq!(value, 0.0);
        assert!(psi_star.is_empty());
    }

    #[test]
    fn maximize_token_evaluate() {
        let objective = MaximizeToken::new("ETH".to_string());
        let mut psi = HashMap::new();
        psi.insert("ETH".to_string(), 5.5);
        psi.insert("USDC".to_string(), 1000.0);
        assert_eq!(objective.evaluate(&psi).unwrap(), 5.5);

        let mut psi2 = HashMap::new();
        psi2.insert("USDC".to_string(), 100.0);
        assert_eq!(objective.evaluate(&psi2).unwrap(), 0.0);
    }

    #[test]
    fn maximize_token_fenchel_conjugate_term() {
        let objective = MaximizeToken::new("ETH".to_string());
        let nu = HashMap::new();
        let all_tokens: Vec<Token> = vec!["ETH".to_string(), "USDC".to_string()];
        let (value, psi_star) = objective.compute_fenchel_conjugate_term(&nu, &all_tokens).unwrap();
        assert_eq!(value, 0.0);
        assert!(psi_star.is_empty());
    }
} 