// cfmm_router_rs/examples/arbitrage_example.rs

use cfmm_router_rs::cfmm::{CFMM, ProductTwoCoinCFMM};
use cfmm_router_rs::objective::{LinearNonnegative};
use cfmm_router_rs::router::Router;
use cfmm_router_rs::types::{Token, Price, CfmrError, NetTradeVec};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CFMM Router Arbitrage Example");

    // 1. Define Tokens
    let eth = "ETH".to_string();
    let usdc = "USDC".to_string();
    let dai = "DAI".to_string();

    // 2. Define Objective Function (e.g., arbitrage against reference prices)
    // We want to maximize profit in terms of USDC (or any numeraire).
    // The LinearNonnegative objective takes a set of prices `p` and maximizes `p^T Ψ`.
    // If Ψ_k is positive, it's a net gain of token k. If negative, it's a net cost.
    let mut reference_prices = HashMap::new();
    reference_prices.insert(eth.clone(), 2000.0); // 1 ETH = 2000 USDC
    reference_prices.insert(usdc.clone(), 1.0);   // 1 USDC = 1 USDC
    reference_prices.insert(dai.clone(), 0.99);   // 1 DAI = 0.99 USDC (example market rate)
    let objective = Box::new(LinearNonnegative::new(reference_prices.clone()));

    // 3. Define CFMMs
    let mut cfmms: Vec<Box<dyn CFMM>> = Vec::new();

    // CFMM 1: ETH/USDC pool (Uniswap V2 style)
    // Pool price: 200_000 USDC / 100 ETH = 2000 USDC/ETH
    let mut reserves1 = HashMap::new();
    reserves1.insert(eth.clone(), 100.0);
    reserves1.insert(usdc.clone(), 200_000.0);
    let cfmm1 = ProductTwoCoinCFMM::new(reserves1, 0.003)?;
    cfmms.push(Box::new(cfmm1));

    // CFMM 2: ETH/DAI pool
    // Pool price: 145_000 DAI / 50 ETH = 2900 DAI/ETH
    let mut reserves2 = HashMap::new();
    reserves2.insert(eth.clone(), 50.0);
    reserves2.insert(dai.clone(), 145_000.0);
    let cfmm2 = ProductTwoCoinCFMM::new(reserves2, 0.003)?;
    cfmms.push(Box::new(cfmm2));
    
    // CFMM 3: USDC/DAI pool
    // Pool price: 100_000 DAI / 100_000 USDC = 1 DAI/USDC (DAI is slightly cheaper in reference_prices)
    let mut reserves3 = HashMap::new();
    reserves3.insert(usdc.clone(), 100_000.0);
    reserves3.insert(dai.clone(), 100_000.0); 
    let cfmm3 = ProductTwoCoinCFMM::new(reserves3, 0.001)?;
    cfmms.push(Box::new(cfmm3));


    // 4. Create Router
    // Max iterations and tolerance for the solver (currently a placeholder)
    let router = Router::new(objective, cfmms, 100, 1e-6);
    println!("Router created with {} CFMMs.", router.all_tokens().len());
    println!("All tokens in router: {:?}", router.all_tokens());

    // 5. Perform Routing
    println!("\nAttempting to route...");
    match router.route() {
        Ok(result) => {
            println!("Routing successful (though solver is a placeholder)!");
            println!("Objective Value (Profit in USDC): {:.2}", result.objective_value);
            println!("Net Trades (Ψ):");
            print_net_trades(&result.net_trades, &reference_prices);

            println!("\nOptimal Dual Variables (ν - internal prices):");
            for (token, price) in result.nu.iter().filter(|(_, &p)| p.abs() > 1e-9) { // Filter out near-zero prices
                println!("  {}: {:.4}", token, price);
            }

            println!("\nIndividual CFMM Trades:");
            for i in 0..result.deltas.len() {
                println!("  CFMM {}:", i + 1);
                if result.deltas[i].is_empty() && result.lambdas[i].is_empty() {
                    println!("    No trade.");
                } else {
                    for (token, amount) in &result.deltas[i] {
                        println!("    Tender (Δ_{}): {} {:.4}", i + 1, token, amount);
                    }
                    for (token, amount) in &result.lambdas[i] {
                        println!("    Receive (Λ_{}): {} {:.4}", i + 1, token, amount);
                    }
                }
            }
        }
        Err(CfmrError::OptimizationError(e)) => {
            println!("Routing failed as expected (solver not implemented): {}", e);
            println!("This example demonstrates API usage. A real solver is needed for actual results.");
        }
        Err(e) => {
            println!("An unexpected error occurred: {:?}", e);
        }
    }

    Ok(())
}

fn print_net_trades(net_trades: &NetTradeVec, prices: &HashMap<Token, Price>) {
    let mut total_value = 0.0;
    for (token, amount) in net_trades {
        if amount.abs() > 1e-9 { // Only print if significant
            let price = prices.get(token).unwrap_or(&0.0);
            println!("  {}: {:.6} (Value @ reference price: {:.2})", token, amount, amount * price);
            total_value += amount * price;
        }
    }
    println!("  ---------------------------------------------------");
    println!("  Total Value of Net Trades @ reference prices: {:.2}", total_value);
    println!("  (This should match the objective value if prices match the objective's numeraire)");
} 