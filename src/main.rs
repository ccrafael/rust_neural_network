use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, VarMap};
use rust_neural_network::Mlp;

const LR: f64 = 0.1;
const EPOCHS: usize = 3000;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // XOR problem: the classic test that requires a non-linear model.
    let xs = Tensor::new(
        &[[-1f32, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        &device,
    )?;
    let ys = Tensor::new(&[-1f32, 1.0, 1.0, -1.0], &device)?.unsqueeze(1)?;

    let vs = VarMap::new();
    let mlp = Mlp::new(&vs, 2, 8, 8, 1, &device, DType::F32)?;
    let mut optim = AdamW::new_lr(vs.all_vars(), LR)?;

    for epoch in 0..EPOCHS {
        let logits = mlp.forward(&xs)?;
        let loss = (logits - &ys)?.sqr()?.mean_all()?;
        if epoch % 300 == 0 {
            println!("epoch {epoch:4}  loss {:.6}", loss.to_scalar::<f32>()?);
        }
        let grads = loss.backward()?;
        optim.step(&grads)?;
    }

    let preds = mlp.forward(&xs)?.round()?;
    println!("\n--- predictions (learned XOR) ---");
    let xs_v = xs.to_vec2::<f32>()?;
    let ys_v = ys.flatten_all()?.to_vec1::<f32>()?;
    let preds_v = preds.flatten_all()?.to_vec1::<f32>()?;
    for ((x, y), p) in xs_v.iter().zip(ys_v).zip(preds_v) {
        println!("inputs {x:?}  target {y:>2.0}  prediction {p:>2.0}");
    }

    Ok(())
}
