# Rust Neural Network Engine

A from-scratch neural network engine implementation in Rust.

## Current Progress

The project currently implements a basic `Tensor` library that supports:

- **Tensor Creation**: Initialize tensors with specific dimensions and data.
- **Summation**: Calculate the sum of all elements in a tensor.
- **Element-wise Multiplication**: Multiply two tensors of the same shape element by element.
- **Dot Product (Matrix Multiplication)**: Perform matrix multiplication for 2D tensors.

## Getting Started

### Prerequisites

You need to have [Rust and Cargo](https://www.rust-lang.org/tools/install) installed on your system.

### Running the Example

To run the included demonstration, navigate to the project directory and use `cargo run`:

```bash
cd rust_neural_network
cargo run
```

## Project Structure

- `src/lib.rs`: Contains the core `Tensor` implementation.
- `src/main.rs`: A demonstration of the tensor operations.
