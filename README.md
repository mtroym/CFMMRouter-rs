# CFMM Router RS

`cfmm_router_rs` is a Rust library for optimal routing of trades across a network of Constant Function Market Makers (CFMMs). It is inspired by the functionality of the Julia library `CFMMRouter.jl`. The primary goal is to find the sequence of trades that maximizes a user-defined objective function, such as arbitrage profit against a set of reference prices or maximizing the amount of a specific token received.

## Overview

The library provides tools to:
1.  Define different types of CFMMs (e.g., constant product pools like Uniswap V2).
2.  Specify objective functions (e.g., linear profit maximization, maximizing a specific token).
3.  Route trades across a collection of CFMMs to achieve the specified objective.

It uses a dual decomposition approach to solve the routing problem, with the L-BFGS-B algorithm (via the `argmin` crate) employed to solve the dual problem.

## Core Concepts

-   **`CFMM` Trait**: An abstraction for any constant function market maker. Implementors of this trait define their specific trading logic and how to solve the arbitrage subproblem for given dual variables (prices).
    -   `ProductTwoCoinCFMM`: A concrete implementation for two-token constant product pools.
-   **`ObjectiveFunction` Trait**: An abstraction for the user's utility function U(Ψ), where Ψ is the net trade vector. The router aims to maximize this function.
    -   `LinearNonnegative`: Maximize `p^T Ψ` for a given price vector `p`.
    -   `MaximizeToken`: Maximize the amount of a specific target token in Ψ.
-   **`Router`**: The main struct that orchestrates the optimization. It takes a set of CFMMs and an objective function, then solves for the optimal dual variables and reconstructs the primal trade solution.
-   **`Solver`**: The underlying optimization algorithm. Currently, this library uses an L-BFGS-B implementation from the `argmin` crate, adapted to handle non-negativity constraints on dual variables via a transformation (`ν_k = exp(α_k)`).

## Features

-   **Abstraction via Traits**: Core components like CFMMs and Objective Functions are defined using traits, allowing for extensibility with new pool types or optimization goals.
-   **Pure Rust Solver**: Utilizes the `argmin` crate for optimization, avoiding complex C/Fortran dependencies for the solver itself (though BLAS linkage is needed for `ndarray-linalg`).
-   **Arbitrage Subproblem Solution**: Includes analytical solutions for common CFMM types (e.g., two-coin constant product).
-   **Example Usage**: Provides an example (`arbitrage_example.rs`) demonstrating how to set up and run the router.

## Project Structure

The library is organized into several modules within the `src` directory:

-   `lib.rs`: The main library crate root, re-exporting key components.
-   `types.rs`: Defines core data structures (e.g., `Token`, `Amount`, `Price`, `Reserves`, `ArbitrageResult`, `RouterResult`, `CfmrError`).
-   `cfmm.rs`: Contains the `CFMM` trait and implementations like `ProductTwoCoinCFMM`.
-   `objective.rs`: Contains the `ObjectiveFunction` trait and implementations like `LinearNonnegative` and `MaximizeToken`.
-   `router.rs`: Defines the `Router` struct and its `route` method, which ties everything together.
-   `solvers.rs`: Implements the optimization logic, including the wrapper for `argmin` and the transformation for handling constrained dual variables.

## Dependencies

Key dependencies include:
-   `argmin`: For the L-BFGS-B optimization algorithm.
-   `argmin-math`: Provides math traits for `argmin`, with `ndarray` as the backend.
-   `ndarray`: For numerical vector and matrix operations.
-   `ndarray-linalg`: For BLAS-powered linear algebra operations used by `argmin`.
    -   This library is configured to link against a BLAS provider. The `Cargo.toml` is currently set up to attempt static linking with **OpenBLAS** (`openblas-static` feature). Ensure OpenBLAS is installed and discoverable on your system. Alternatively, for macOS, you might consider using the `accelerate` feature by changing `ndarray-linalg`'s features in `Cargo.toml` and adding `accelerate-src` as a dependency.
-   `approx`: For floating-point comparisons in tests.

## Building the Project

1.  **Install Rust**: If you haven't already, install Rust and Cargo from [rustup.rs](https://rustup.rs/).
2.  **BLAS Dependency**:
    -   If using `openblas-static` (current default in `Cargo.toml` after your last change): Ensure OpenBLAS (including static libraries and headers) is installed on your system. You might need to set environment variables like `OPENBLAS_PATH` or `OPENBLAS_LIB_DIR` and `OPENBLAS_INCLUDE_DIR` if Cargo cannot find it automatically.
    -   If you switch to `accelerate` on macOS: Add `accelerate-src = "0.3.2"` to `[dependencies]` and change `ndarray-linalg` features to `["accelerate"]`. This usually works out-of-the-box on macOS.
3.  **Build**: Navigate to the `cfmm_router_rs` directory and run:
    ```bash
    cargo build
    ```
    For a release build:
    ```bash
    cargo build --release
    ```

## Running Examples

To run the provided arbitrage example:
```bash
cargo run --example arbitrage_example
```

## Running Tests

To run the unit tests:
```bash
cargo test
```

## Basic Usage

Here's a conceptual overview of how to use the router (see `examples/arbitrage_example.rs` for a runnable version):

```rust
use cfmm_router_rs::cfmm::{ProductTwoCoinCFMM, CFMM};
use cfmm_router_rs::objective::LinearNonnegative;
use cfmm_router_rs::router::Router;
use cfmm_router_rs::types::{Token, Price, Reserves}; // Adjust imports as needed
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define tokens
    let token_a = "TOKEN_A".to_string();
    let token_b = "TOKEN_B".to_string();

    // 2. Create CFMMs
    let mut reserves1 = HashMap::new();
    reserves1.insert(token_a.clone(), 1000.0);
    reserves1.insert(token_b.clone(), 1000.0);
    let cfmm1 = ProductTwoCoinCFMM::new(reserves1, 0.003)?;
    
    let cfmms: Vec<Box<dyn CFMM>> = vec![Box::new(cfmm1)];

    // 3. Define an objective function
    let mut prices = HashMap::new();
    prices.insert(token_a.clone(), 1.0);
    prices.insert(token_b.clone(), 1.01); // Target price for Token B slightly higher
    let objective = Box::new(LinearNonnegative::new(prices));

    // 4. Create and run the router
    let router = Router::new(objective, cfmms, 100, 1e-6);
    match router.route() {
        Ok(result) => {
            println!("Objective value: {}", result.objective_value);
            println!("Net trades: {:?}", result.net_trades);
            // Process other results...
        }
        Err(e) => {
            eprintln!("Routing error: {:?}", e);
        }
    }
    Ok(())
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the [MIT License](LICENSE.txt)