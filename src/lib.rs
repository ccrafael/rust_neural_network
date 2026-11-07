use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder, VarMap};

/// A multilayer perceptron with two hidden layers and tanh activations.
pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
    head: Linear,
}

impl Mlp {
    pub fn new(
        vs: &VarMap,
        in_dim: usize,
        hidden1: usize,
        hidden2: usize,
        out_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(vs, dtype, device);
        Ok(Self {
            ln1: linear(in_dim, hidden1, vb.pp("ln1"))?,
            ln2: linear(hidden1, hidden2, vb.pp("ln2"))?,
            head: linear(hidden2, out_dim, vb.pp("head"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?.tanh()?;
        let xs = self.ln2.forward(&xs)?.tanh()?;
        self.head.forward(&xs)
    }
}
