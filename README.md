# Rust Neural Network Engine

Neural network examples in Rust built on [candle-core](https://github.com/huggingface/candle) — no hand-rolled tensor math.

## What's Here

- `src/lib.rs`: `Mlp`, a multilayer perceptron (2 hidden layers, tanh activations) built with `candle-nn` linear layers and a `VarMap` for its parameters.
- `src/main.rs`: Trains the MLP on the XOR problem using MSE loss and AdamW (`candle_nn::AdamW`), then prints the learned predictions.

## Getting Started

### Prerequisites

You need to have [Rust and Cargo](https://www/rust-lang.org/tools/install) installed on your system.

### Running the Example

```bash
cd rust_neural_network
cargo run
```

Expected output: the loss drops to ~0 and the network classifies all four XOR inputs correctly.
