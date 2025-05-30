use crate::cfmm::CFMM;
use crate::objective::ObjectiveFunction;
use crate::types::{
    Token, DualVariables, RouterResult, Result, CfmrError,
    Delta, Lambda, get_or_zero, sum_all_net_flows,
};
use crate::solvers::{self, OptimizationProblem};
use std::collections::{HashMap, HashSet};

/// The Router is responsible for finding the optimal set of trades across a list of CFMMs
/// to maximize a given objective function.
pub struct Router {
    objective: Box<dyn ObjectiveFunction>,
    cfmms: Vec<Box<dyn CFMM>>,
    all_tokens: Vec<Token>, // Cached list of all unique tokens across all CFMMs
    max_iterations: usize, // Max iterations for the optimization solver
    tolerance: f64, // Convergence tolerance for the optimization solver
}

impl Router {
    pub fn new(
        objective: Box<dyn ObjectiveFunction>,
        cfmms: Vec<Box<dyn CFMM>>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        let mut token_set = HashSet::new();
        for cfmm in &cfmms {
            for token in cfmm.tokens() {
                token_set.insert(token);
            }
        }
        let mut all_tokens: Vec<Token> = token_set.into_iter().collect();
        // Sort tokens for canonical ordering, important for vector representations of nu
        all_tokens.sort();

        Self {
            objective,
            cfmms,
            all_tokens,
            max_iterations,
            tolerance,
        }
    }

    /// Executes the routing algorithm to find the optimal trades.
    ///
    /// This involves minimizing the dual function g(ν) = (-U)*(-ν) + Σ arb_i(A_i^T ν)
    /// using an optimization algorithm (e.g., L-BFGS-B).
    pub fn route(&self) -> Result<RouterResult> {
        if self.cfmms.is_empty() {
            return Err(CfmrError::InvalidInput("No CFMMs provided to the router.".to_string()));
        }

        // Initial guess for dual variables ν (e.g., all zeros or based on objective prices).
        let initial_nu: DualVariables = self.all_tokens.iter().map(|t| (t.clone(), 1.0)).collect(); 
        // Using 1.0 as a generic starting point. Could be 0.0 or derived from objective.

        // Define the optimization problem (dual function g(ν) and its gradient ∇g(ν)).
        let opt_problem = RouterOptimizationProblem {
            objective: self.objective.as_ref(),
            cfmms: &self.cfmms,
            all_tokens: &self.all_tokens,
        };

        // Solve for optimal ν* using a numerical optimizer.
        let optimal_nu = solvers::minimize_scalar_function(
            &opt_problem,
            initial_nu,
            &self.all_tokens,
            self.max_iterations as u64,
            self.tolerance,
        )?;

        // With optimal ν*, reconstruct the primal solution (optimal trades Δ*_i, Λ*_i).
        let mut optimal_deltas: Vec<Delta> = Vec::with_capacity(self.cfmms.len());
        let mut optimal_lambdas: Vec<Lambda> = Vec::with_capacity(self.cfmms.len());

        for cfmm in &self.cfmms {
            let arbitrage_result = cfmm.solve_arbitrage_subproblem(&optimal_nu)?;
            optimal_deltas.push(arbitrage_result.delta);
            optimal_lambdas.push(arbitrage_result.lambda);
        }

        // Calculate the net trade vector Ψ* = Σ A_i (Λ*_i - Δ*_i).
        let net_trades = sum_all_net_flows(&optimal_lambdas, &optimal_deltas, &self.all_tokens);

        // Calculate the objective value U(Ψ*).
        let objective_value = self.objective.evaluate(&net_trades)?;

        Ok(RouterResult {
            objective_value,
            net_trades,
            deltas: optimal_deltas,
            lambdas: optimal_lambdas,
            nu: optimal_nu,
        })
    }

    pub fn all_tokens(&self) -> &[Token] {
        &self.all_tokens
    }
}

/// Helper struct to implement the `OptimizationProblem` trait for the Router.
/// This represents the dual function g(ν) and its gradient ∇g(ν).
struct RouterOptimizationProblem<'a> {
    objective: &'a dyn ObjectiveFunction,
    cfmms: &'a [Box<dyn CFMM>],
    all_tokens: &'a [Token], // Canonical order for nu_vec
}

impl<'a> OptimizationProblem for RouterOptimizationProblem<'a> {
    /// Calculates g(ν) = (-U)*(-ν) + Σ_i arb_i(A_i^T ν)
    /// and ∇g(ν) = -Ψ* + Σ_i A_i (Λ*_i - Δ*_i)
    fn evaluate(&self, nu_vec: &[f64], token_order: &[Token]) -> Result<(f64, Vec<f64>)> {
        if token_order != self.all_tokens {
            // This check ensures consistency if token_order might vary.
            // However, the solver is initialized with self.all_tokens, so they should match.
            return Err(CfmrError::InvalidInput("Token order mismatch in RouterOptimizationProblem".to_string()));
        }
        let nu_map = self.vec_to_nu_map(nu_vec, token_order);

        // 1. Calculate the Fenchel conjugate term and its contribution to the gradient.
        let (fenchel_term_value, psi_star_for_gradient) = self
            .objective
            .compute_fenchel_conjugate_term(&nu_map, self.all_tokens)?;

        // Initialize g(ν) and ∇g(ν)
        let mut g_nu = fenchel_term_value;
        // Gradient part from Fenchel term: -Ψ*
        // Convert -Ψ* (NetTradeVec) to a vector ordered by self.all_tokens.
        let mut grad_g_nu_map: HashMap<Token, f64> = self.all_tokens.iter().map(|t| (t.clone(), 0.0)).collect();
        for (token, amount) in psi_star_for_gradient {
            *grad_g_nu_map.entry(token).or_insert(0.0) -= amount; 
        }

        // 2. Calculate sum of arbitrage profits and their gradient contributions.
        for cfmm in self.cfmms {
            let arbitrage_result = cfmm.solve_arbitrage_subproblem(&nu_map)?;
            g_nu += arbitrage_result.profit; // arb_i(A_i^T ν)

            // Add A_i (Λ*_i - Δ*_i) to the gradient.
            // The A_i matrix (local to global token mapping) is implicitly handled
            // as Δ and Λ are in terms of global token names.
            let cfmm_tokens = cfmm.tokens(); // Tokens relevant to this specific CFMM

            for token_name in &cfmm_tokens {
                let lambda_val = get_or_zero(&arbitrage_result.lambda, token_name);
                let delta_val = get_or_zero(&arbitrage_result.delta, token_name);
                let net_flow = lambda_val - delta_val;
                *grad_g_nu_map.entry(token_name.clone()).or_insert(0.0) += net_flow;
            }
        }
        
        let grad_g_nu_vec = self.nu_map_to_vec(&grad_g_nu_map, self.all_tokens);

        Ok((g_nu, grad_g_nu_vec))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfmm::{ProductTwoCoinCFMM};
    use crate::objective::{LinearNonnegative};
    use std::collections::HashMap;
    use approx::assert_abs_diff_eq; // For floating point comparisons

    fn setup_simple_router() -> Router {
        let mut prices = HashMap::new();
        prices.insert("ETH".to_string(), 2000.0);
        prices.insert("USDC".to_string(), 1.0);
        let objective = Box::new(LinearNonnegative::new(prices));

        let mut reserves1 = HashMap::new();
        reserves1.insert("ETH".to_string(), 100.0);
        reserves1.insert("USDC".to_string(), 200000.0); // Pool price 2000
        let cfmm1 = Box::new(ProductTwoCoinCFMM::new(reserves1, 0.003).unwrap());

        let mut reserves2 = HashMap::new();
        reserves2.insert("ETH".to_string(), 50.0);
        reserves2.insert("DAI".to_string(), 150000.0); // Pool price 3000 for ETH/DAI
        let cfmm2 = Box::new(ProductTwoCoinCFMM::new(reserves2, 0.003).unwrap());
        
        // Note: DAI is not in the objective prices, so its price is effectively 0 for LinearNonnegative objective.

        Router::new(objective, vec![cfmm1, cfmm2], 100, 1e-6)
    }

    #[test]
    fn router_creation() {
        let router = setup_simple_router();
        assert_eq!(router.cfmms.len(), 2);
        let mut expected_tokens = vec!["ETH".to_string(), "USDC".to_string(), "DAI".to_string()];
        let mut actual_tokens = router.all_tokens.clone();
        expected_tokens.sort();
        actual_tokens.sort();
        assert_eq!(actual_tokens, expected_tokens);
    }

    #[test]
    fn router_route_placeholder_solver() {
        let router = setup_simple_router();
        let result = router.route();
        // With a real solver, we expect it to run, even if no optimal arbitrage is found.
        // It should return Ok, potentially with a zero objective_value if no profit.
        assert!(result.is_ok(), "router.route() should return Ok, but was {:?}", result);
    }

    // More detailed tests will require a mock or simple solver, 
    // or testing components like RouterOptimizationProblem.evaluate directly.

    #[test]
    fn router_optimization_problem_evaluate_no_cfmms() {
        // Test with LinearNonnegative objective (Fenchel term = 0, grad = 0)
        let prices = HashMap::new(); // No specific prices means objective is effectively 0 for all tokens.
        let objective = Box::new(LinearNonnegative::new(prices));
        let all_tokens = vec!["ETH".to_string(), "USDC".to_string()];
        
        let problem = RouterOptimizationProblem {
            objective: objective.as_ref(),
            cfmms: &[],
            all_tokens: &all_tokens,
        };

        let nu_vec = vec![1.0, 1.0]; // nu_eth = 1, nu_usdc = 1
        let (g_val, grad_g_vec) = problem.evaluate(&nu_vec, &all_tokens).unwrap();

        assert_eq!(g_val, 0.0); // Only Fenchel term from LinearNonnegative
        assert_eq!(grad_g_vec, vec![0.0, 0.0]); // Grad from LinearNonnegative Fenchel is 0
    }

    #[test]
    fn router_optimization_problem_evaluate_with_cfmm_no_arb() {
        let mut prices = HashMap::new();
        prices.insert("ETH".to_string(), 2000.0);
        prices.insert("USDC".to_string(), 1.0);
        let objective = Box::new(LinearNonnegative::new(prices));

        let mut reserves1 = HashMap::new();
        reserves1.insert("ETH".to_string(), 100.0);
        reserves1.insert("USDC".to_string(), 200000.0); // Pool price 2000
        let cfmm1 = Box::new(ProductTwoCoinCFMM::new(reserves1, 0.003).unwrap());
        let cfmms: Vec<Box<dyn CFMM>> = vec![cfmm1];
        
        let all_tokens_set: HashSet<Token> = cfmms.iter().flat_map(|c| c.tokens()).collect();
        let mut all_tokens: Vec<Token> = all_tokens_set.into_iter().collect();
        all_tokens.sort(); // Ensure consistent order
        
        let problem = RouterOptimizationProblem {
            objective: objective.as_ref(),
            cfmms: &cfmms,
            all_tokens: &all_tokens,
        };

        // Nu matches pool price, so no arbitrage profit from cfmm1
        let mut nu_map = HashMap::new();
        nu_map.insert("ETH".to_string(), 2000.0);
        nu_map.insert("USDC".to_string(), 1.0);
        let nu_vec = problem.nu_map_to_vec(&nu_map, &all_tokens);

        let (g_val, grad_g_vec) = problem.evaluate(&nu_vec, &all_tokens).unwrap();

        // Fenchel term (LinearNonnegative) = 0
        // Arb profit from cfmm1 = 0 (prices match)
        assert_eq!(g_val, 0.0);
        // Grad from Fenchel = [0,0]
        // Grad from cfmm1 (Λ*-Δ*) = [0,0] because no arb
        let expected_grad_map = all_tokens.iter().map(|t| (t.clone(),0.0)).collect::<HashMap<_,_>>();
        let expected_grad_vec = problem.nu_map_to_vec(&expected_grad_map, &all_tokens);
        assert_eq!(grad_g_vec, expected_grad_vec);
    }

    #[test]
    fn router_optimization_problem_evaluate_with_cfmm_with_arb() {
        let mut obj_prices = HashMap::new();
        // Market prices: ETH = 2050 USDC, USDC = 1 (ETH is overvalued in market vs pool)
        obj_prices.insert("ETH".to_string(), 2050.0);
        obj_prices.insert("USDC".to_string(), 1.0);
        let objective = Box::new(LinearNonnegative::new(obj_prices));

        let mut reserves1 = HashMap::new();
        reserves1.insert("ETH".to_string(), 1000.0); // CFMM has ETH
        reserves1.insert("USDC".to_string(), 2000000.0); // CFMM has USDC, price 2000
        let cfmm1 = Box::new(ProductTwoCoinCFMM::new(reserves1, 0.003).unwrap());
        let cfmms: Vec<Box<dyn CFMM>> = vec![cfmm1];
        
        let all_tokens_set: HashSet<Token> = cfmms.iter().flat_map(|c| c.tokens()).collect();
        let mut all_tokens: Vec<Token> = all_tokens_set.into_iter().collect();
        all_tokens.sort(); // Should be ["ETH", "USDC"]

        let problem = RouterOptimizationProblem {
            objective: objective.as_ref(),
            cfmms: &cfmms,
            all_tokens: &all_tokens,
        };

        // Dual variables matching the objective prices where arbitrage should occur
        // This setup matches the cfmm::tests::solve_arb_sell_eth_for_usdc scenario (after token sorting).
        // In that scenario (nu_eth=2050, nu_usdc=1), the optimal trade is selling USDC for ETH.
        let nu_vec = if all_tokens[0] == "ETH" && all_tokens[1] == "USDC" {
            vec![2050.0, 1.0] // nu_eth = 2050, nu_usdc = 1
        } else if all_tokens[0] == "USDC" && all_tokens[1] == "ETH" {
            vec![1.0, 2050.0] // nu_usdc = 1, nu_eth = 2050
        } else {
            panic!("Unexpected token order in test");
        };
        
        let (g_val, grad_g_vec) = problem.evaluate(&nu_vec, &all_tokens).unwrap();
        
        // Values derived from actual test output for the corresponding CFMM subproblem
        assert_abs_diff_eq!(g_val, 238.46891227356173, epsilon = 1e-7);

        // Expected net flows (Lambda* - Delta*) corresponding to the above g_val
        let eth_idx = all_tokens.iter().position(|t| t == "ETH").unwrap();
        let usdc_idx = all_tokens.iter().position(|t| t == "USDC").unwrap();

        assert_abs_diff_eq!(grad_g_vec[eth_idx], 10.78546701214293, epsilon = 1e-7);
        assert_abs_diff_eq!(grad_g_vec[usdc_idx], -21871.738462619447, epsilon = 1e-7);
    }
} 