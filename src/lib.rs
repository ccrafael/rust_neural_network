pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(data.len(), expected_size, "Data size does not match shape");
        Tensor { data, shape }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensors must have the same shape for element-wise multiplication");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Dot product (matrix multiplication) currently only supports 2D tensors");
        assert_eq!(other.shape.len(), 2, "Dot product (matrix multiplication) currently only supports 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Incompatible shapes for matrix multiplication");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Tensor::new(result_data, vec![m, n])
    }

    pub fn print(&self) {
        println!("Tensor(shape: {:?}, data: {:?})", self.shape, self.data);
    }
}
