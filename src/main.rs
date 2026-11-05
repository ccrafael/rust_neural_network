use rust_neural_network::Tensor;

fn main() {
    // Test 1: Square matrices (2x2 * 2x2)
    println!("--- Test 1: Square matrices (2x2 * 2x2) ---");
    let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("Tensor 1:");
    tensor1.print();
    println!("Tensor 2:");
    tensor2.print();

    println!("\nElement-wise multiplication:");
    let mul_result = tensor1.mul(&tensor2);
    mul_result.print();

    println!("\nMatrix multiplication (dot product):");
    let dot_result = tensor1.dot(&tensor2);
    dot_result.print();

    // Test 2: Non-square matrices (2x3 * 3x2)
    println!("\n--- Test 2: Non-square matrices (2x3 * 3x2) ---");
    // 2x3 matrix
    let tensor3 = Tensor::new(vec![
        1.0, 2.0, 3.0, 
        4.0, 5.0, 6.0
    ], vec![2, 3]);
    
    // 3x2 matrix
    let tensor4 = Tensor::new(vec![
        7.0, 8.0, 
        9.0, 10.0, 
        11.0, 12.0
    ], vec![3, 2]);

    println!("Tensor 3 (2x3):");
    tensor3.print();
    println!("Tensor 4 (3x2):");
    tensor4.print();

    println!("\nMatrix multiplication (dot product):");
    let dot_non_square_result = tensor3.dot(&tensor4);
    dot_non_square_result.print();
    // Expected result:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [7 + 18 + 33, 8 + 20 + 36] = [58, 64]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [28 + 45 + 66, 32 + 50 + 72] = [139, 154]
}
