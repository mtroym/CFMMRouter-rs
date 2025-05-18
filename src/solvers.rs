// cfmm_router_rs/src/solvers.rs

use crate::types::{DualVariables, Result, CfmrError, Token}; // Removed Price, not used directly here
use std::collections::HashMap;

// argmin related imports
use argmin::core::{Error, Executor, CostFunction, Gradient, ArgminError}; // Removed State import, and now IterState
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS as QuasiNewtonLBFGS; // More common path for LBFGS

// ndarray for vector operations with argmin
use ndarray::{Array1}; // Removed ArrayView1

/// Represents the function g(ν) to be minimized, and its gradient ∇g(ν).
pub trait OptimizationProblem: Send + Sync {
    /// Evaluates the function g(ν) and its gradient ∇g(ν) at a given point ν.
    ///
    /// # Arguments
    /// * `nu_vec`: The current dual variables ν, represented as a flat Vec<f64>.
    ///             The order is determined by `token_order`.
    /// * `token_order`: A slice defining the canonical order of tokens for nu_vec.
    ///
    /// # Returns
    /// A tuple `(value, gradient_vec)`:
    ///   - `value`: The value of the function g(ν).
    ///   - `gradient_vec`: The gradient ∇g(ν) as a flat Vec<f64>, in the same order as `token_order`.
    fn evaluate(&self, nu_vec: &[f64], token_order: &[Token]) -> Result<(f64, Vec<f64>)>;

    /// Converts a flat Vec<f64> of dual variables to a HashMap<Token, f64>.
    fn vec_to_nu_map(&self, nu_vec: &[f64], token_order: &[Token]) -> DualVariables {
        token_order.iter().zip(nu_vec.iter()).map(|(t, &v)| (t.clone(), v)).collect()
    }

    /// Converts a HashMap<Token, f64> of dual variables to a flat Vec<f64>.
    fn nu_map_to_vec(&self, nu_map: &DualVariables, token_order: &[Token]) -> Vec<f64> {
        token_order.iter().map(|t| *nu_map.get(t).unwrap_or(&0.0)).collect()
    }
}

// Wrapper struct to adapt OptimizationProblem to Operator using variable transformation
// We optimize for alpha, where nu_k = exp(alpha_k)
struct TransformedArgminProblem<'a> {
    inner_problem: &'a dyn OptimizationProblem,
    token_order: &'a [Token],
    // Cache for nu and grad_nu to avoid recomputing if evaluate is called by both apply and gradient
    // For LBFGS, apply and gradient are typically called separately for the same point.
}

impl<'a> TransformedArgminProblem<'a> {
    fn transform_alpha_to_nu(&self, alpha_vec: &Array1<f64>) -> Vec<f64> {
        alpha_vec.iter().map(|&alpha_k| alpha_k.exp()).collect()
    }
}

// Implement argmin::core::CostFunction trait
impl<'a> CostFunction for TransformedArgminProblem<'a> {
    type Param = Array1<f64>; 
    type Output = f64;

    fn cost(&self, alpha_array: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let nu_vec = self.transform_alpha_to_nu(alpha_array);
        match self.inner_problem.evaluate(&nu_vec, self.token_order) {
            Ok((value, _gradient)) => Ok(value),
            Err(e) => {
                eprintln!("Error in CostFunction cost (evaluating objective): {:?}", e);
                Err(Error::from(e))
            }
        }
    }
}

impl<'a> Gradient for TransformedArgminProblem<'a> {
    type Param = Array1<f64>; 
    type Gradient = Array1<f64>;
    // Jacobian and Hessian are not base associated types of Gradient trait directly
    // They are inferred or handled if jacobian()/hessian() methods are implemented.

    fn gradient(&self, alpha_array: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
        let nu_vec = self.transform_alpha_to_nu(alpha_array);
        match self.inner_problem.evaluate(&nu_vec, self.token_order) {
            Ok((_value, grad_nu_vec)) => {
                if grad_nu_vec.len() != alpha_array.len() {
                    let err_msg = "Gradient dimension mismatch in transformation".to_string();
                    return Err(Error::from(ArgminError::ConditionViolated{ text: err_msg }));
                }
                let grad_alpha_vec: Vec<f64> = grad_nu_vec.iter().zip(nu_vec.iter())
                    .map(|(&dn_k, &n_k)| dn_k * n_k)
                    .collect();
                Ok(Array1::from_vec(grad_alpha_vec))
            }
            Err(e) => {
                 eprintln!("Error in Gradient gradient (evaluating gradient): {:?}", e);
                 Err(Error::from(e))
            }
        }
    }
}

pub fn minimize_scalar_function(
    problem: &dyn OptimizationProblem,
    initial_nu_map: DualVariables,
    token_order: &[Token],
    max_iterations: u64, // argmin uses u64 for iterations
    _pg_tolerance: f64, // LBFGS in argmin uses gradient norm tolerance
) -> Result<DualVariables> {
    let n = token_order.len();
    if n == 0 {
        return Ok(HashMap::new());
    }

    // Initial nu_vec and transform to initial alpha_vec
    // nu_k = exp(alpha_k) => alpha_k = ln(nu_k)
    // Handle nu_k = 0 or small nu_k. Let's use a small epsilon.
    let epsilon = 1e-10; // Small positive number to avoid ln(0)
    let initial_nu_vec = problem.nu_map_to_vec(&initial_nu_map, token_order);
    let initial_alpha_vec: Vec<f64> = initial_nu_vec
        .iter()
        .map(|&nu_k| nu_k.max(epsilon).ln())
        .collect();
    let initial_alpha_array = Array1::from_vec(initial_alpha_vec);

    // Instantiate the problem for argmin
    let transformed_problem = TransformedArgminProblem {
        inner_problem: problem,
        token_order,
    };

    // Set up the LBFGS solver
    // The LBFGS solver in argmin is often found in argmin::solver::quasinewton::LBFGS
    // It might require a line search. MoreThuenteLineSearch is a common choice.
    let linesearch = MoreThuenteLineSearch::new();
    // LBFGS parameters: `m` is the number of past gradients to store.
    let m_corrections = 7; // Common default for LBFGS history size
    let solver = QuasiNewtonLBFGS::new(linesearch, m_corrections);
    
    // Run the executor
    let res = Executor::new(transformed_problem, solver)
        // Let the compiler infer the type of state
        .configure(|state| state.param(initial_alpha_array).max_iters(max_iterations))
        // .add_observer(ArgminSlogLogger::term(), ObserverMode::Always) // Optional: for logging
        .run();

    match res {
        Ok(final_state) => {
            // final_state.state() returns &IterState. best_param is Option<P>.
            // Clone the Option<P> to take ownership if P is Clone (Array1 is Clone).
            if let Some(best_alpha_array) = final_state.state().best_param.clone() {
                let best_nu_vec: Vec<f64> = best_alpha_array.iter().map(|&alpha_k| alpha_k.exp()).collect();
                Ok(problem.vec_to_nu_map(&best_nu_vec, token_order))
            } else {
                 Err(CfmrError::OptimizationError(
                    "argmin LBFGS finished but no best parameters found.".to_string()
                ).into())
            }
        }
        Err(e) => {
            Err(CfmrError::OptimizationError(format!(
                "argmin LBFGS optimization failed: {}",
                e
            )).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Token, Result as CfmrResult};
    use std::collections::HashMap;
    use ndarray::array; // For creating arrays in tests if needed

    struct SimpleQuadraticProblem {
        a: f64,
        b: f64,
        token_x: Token,
        token_y: Token,
    }

    impl OptimizationProblem for SimpleQuadraticProblem {
        fn evaluate(&self, nu_vec: &[f64], token_order: &[Token]) -> CfmrResult<(f64, Vec<f64>)> {
            let nu_map = self.vec_to_nu_map(nu_vec, token_order);
            let x = *nu_map.get(&self.token_x).unwrap_or(&0.0);
            let y = *nu_map.get(&self.token_y).unwrap_or(&0.0);

            // Objective: (x-a)^2 + (y-b)^2
            // If we transform: x = exp(alpha_x), y = exp(alpha_y)
            // The argmin problem will see f(alpha_x, alpha_y) = (exp(alpha_x)-a)^2 + (exp(alpha_y)-b)^2
            // The `evaluate` here is still for g(nu)
            let value = (x - self.a).powi(2) + (y - self.b).powi(2);
            
            let mut grad_map = HashMap::new();
            grad_map.insert(self.token_x.clone(), 2.0 * (x - self.a)); // d/dx
            grad_map.insert(self.token_y.clone(), 2.0 * (y - self.b)); // d/dy
            
            let grad_vec = self.nu_map_to_vec(&grad_map, token_order);
            Ok((value, grad_vec))
        }
    }

    #[test]
    fn test_simple_quadratic_problem_evaluation() {
        let token_x = "X".to_string();
        let token_y = "Y".to_string();
        let problem = SimpleQuadraticProblem { a: 3.0, b: 4.0, token_x: token_x.clone(), token_y: token_y.clone() };
        let token_order = vec![token_x.clone(), token_y.clone()];
        
        let nu_vec = vec![1.0, 2.0]; // nu_x=1, nu_y=2
        let (value, grad_vec) = problem.evaluate(&nu_vec, &token_order).unwrap();
        assert_eq!(value, 8.0); // (1-3)^2 + (2-4)^2 = 4+4=8
        assert_eq!(grad_vec, vec![-4.0, -4.0]); // 2(1-3), 2(2-4)
    }
    
    #[test]
    fn test_transformed_argmin_problem_interface() {
        let token_x = "X".to_string();
        let token_y = "Y".to_string();
        let inner_problem = SimpleQuadraticProblem { a: 3.0, b: 4.0, token_x: token_x.clone(), token_y: token_y.clone() };
        let token_order = vec![token_x.clone(), token_y.clone()];

        let transformed_problem = TransformedArgminProblem {
            inner_problem: &inner_problem,
            token_order: &token_order,
        };

        // Test with alpha values. Let alpha_x = ln(1), alpha_y = ln(2)
        // So nu_x = 1, nu_y = 2
        let alpha_param = Array1::from_vec(vec![1.0f64.ln(), 2.0f64.ln()]);
        
        // Test CostFunction::cost
        let cost = transformed_problem.cost(&alpha_param).unwrap();
        assert_eq!(cost, 8.0); // Should be (1-3)^2 + (2-4)^2 = 8.0

        // Test Gradient::gradient
        // grad_nu_x = 2*(1-3) = -4
        // grad_nu_y = 2*(2-4) = -4
        // nu_x = 1, nu_y = 2
        // grad_alpha_x = -4 * 1 = -4
        // grad_alpha_y = -4 * 2 = -8
        let grad_alpha = transformed_problem.gradient(&alpha_param).unwrap();
        let expected_grad_alpha = array![-4.0, -8.0];
        assert!(grad_alpha.abs_diff_eq(&expected_grad_alpha, 1e-9));
    }


    #[test]
    fn test_minimize_quadratic_unconstrained_positive_target() {
        // Target (a,b) = (3,4). Since nu_k = exp(alpha_k) > 0, this should be found.
        let token_x = "X".to_string();
        let token_y = "Y".to_string();
        let problem = SimpleQuadraticProblem { a: 3.0, b: 4.0, token_x: token_x.clone(), token_y: token_y.clone() };
        let token_order = vec![token_x.clone(), token_y.clone()];

        let mut initial_nu = HashMap::new();
        // Start somewhat away, e.g., nu = (1,1) => alpha = (0,0)
        initial_nu.insert(token_x.clone(), 1.0);
        initial_nu.insert(token_y.clone(), 1.0);

        let result_nu_map = minimize_scalar_function(&problem, initial_nu, &token_order, 200, 1e-6).unwrap();

        let final_x = *result_nu_map.get(&token_x).unwrap();
        let final_y = *result_nu_map.get(&token_y).unwrap();

        // Check if nu values are close to (3,4)
        assert!((final_x - 3.0).abs() < 1e-3, "x is {}, expected 3.0", final_x);
        assert!((final_y - 4.0).abs() < 1e-3, "y is {}, expected 4.0", final_y);
    }

    #[test]
    fn test_minimize_quadratic_constrained_by_transformation() {
        // Target (a,b) = (3, -2). True minimum is at nu_y = -2.
        // But nu_y = exp(alpha_y) must be > 0.
        // So the optimization should find nu_y close to 0.
        // Objective: (nu_x - 3)^2 + (nu_y - (-2))^2
        // Expected solution with nu_y >= 0 is nu_x = 3, nu_y = epsilon (very small positive)
        let token_x = "X".to_string();
        let token_y = "Y".to_string();
        let problem = SimpleQuadraticProblem { a: 3.0, b: -2.0, token_x: token_x.clone(), token_y: token_y.clone() };
        let token_order = vec![token_x.clone(), token_y.clone()];

        let mut initial_nu = HashMap::new();
        initial_nu.insert(token_x.clone(), 1.0); // alpha_x = 0
        initial_nu.insert(token_y.clone(), 1.0); // alpha_y = 0

        let result_nu_map = minimize_scalar_function(&problem, initial_nu, &token_order, 200, 1e-6).unwrap();

        let final_x = *result_nu_map.get(&token_x).unwrap();
        let final_y = *result_nu_map.get(&token_y).unwrap();
        
        println!("Constrained test: final_x = {}, final_y = {}", final_x, final_y);

        assert!((final_x - 3.0).abs() < 1e-3, "x is {}, expected 3.0", final_x);
        // final_y should be small and positive, effectively our "0" due to exp transform.
        // The actual value depends on how close alpha_y can get to -infinity.
        assert!(final_y > 0.0 && final_y < 1e-2, "y is {}, expected ~0 (small positive)", final_y);
    }
} 