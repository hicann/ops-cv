# Quantization Introduction

Quantization is widely used in deep learning models, especially during the inference process. Through quantization, models can run more efficiently on hardware, reducing the consumption of computational resources and accelerating the inference process, while also reducing the storage requirements of the model.

CANN operator quantization refers to the calculation process of converting the input Tensor of matrix (cube) operators such as Matmul in neural networks from high bit to low bit, while generating the corresponding quantization parameter scale. When the low-bit cube calculation is completed, the low-bit value can be converted back to the high-bit value through the quantization parameter scale, thereby ensuring the correctness of the overall calculation result (the effect is approximately equivalent to direct high-bit calculation) and effectively improving computational efficiency.

- Static quantization: Uses pre-determined quantization parameters for quantization. In inference scenarios, quantization of weight generally uses static quantization, and the quantization operator performance will be better.
- Dynamic quantization: Uses input data to calculate quantization parameters online for quantization. In inference scenarios, quantization of activation generally uses dynamic quantization, which can better adapt to data changes and has higher accuracy. In training scenarios, dynamic quantization is also generally used to improve quantization accuracy. Note that dynamic quantization will have slightly worse quantization operator performance because quantization parameters are generated online.

## Quantization Mode

Quantization mode (also known as quantization granularity) refers to using different quantization calculation levels for different input Tensors of the operator. Common quantization calculation modes include:

> Note:
>
>- m, n, k variables represent the sizes of different axes of Tensor calculation.
>- Left matrix and right matrix refer to the two input Tensors used for matrix multiplication calculation in the cube operator. Generally, the left matrix represents activation and the right matrix represents weight. Please understand and use according to the actual situation.

- pertensor quantization (abbreviated as T quantization): The quantization object can be either the left matrix or the right matrix. Each Tensor shares the same quantization parameter.

  Assume the left matrix shape is (m, k) and the right matrix shape is (k, n). k is the reduce axis. The generated quantization parameter shape is (1, ).
<!--
  ![Schematic Diagram](../figures/pertensor量化.png)
-->
- perchannel quantization (abbreviated as C quantization): The quantization object is the right matrix. Each channel uses independent quantization parameters.

  Assume the right matrix shape is (k, n). k is the reduce axis. The generated quantization parameter shape is (n, ).
<!--
  ![Schematic Diagram](../figures/perchannel量化.png)
-->
- pertoken quantization (abbreviated as K quantization): The quantization object is the left matrix. Each token uses independent quantization parameters.

  Assume the left matrix shape is (m, k). k is the reduce axis. The generated quantization parameter shape is (m, ).
<!--
  ![Schematic Diagram](../figures/pertoken量化.png)
-->
- pergroup quantization (abbreviated as G quantization): The quantization object can be either the left matrix or the right matrix. Data is grouped on the reduce axis, and each group uses independent quantization parameters.
  - Assume the left matrix shape is (m, k). k is the reduce axis. Group on the k axis with group size gs. The generated quantization parameter shape is (m, k/gs).
  - Assume the right matrix shape is (k, n). k is the reduce axis. Group on the k axis with group size gs. The generated quantization parameter shape is (k/gs, n).
<!--
  ![Schematic Diagram](../figures/pergroup量化.png)
-->
- perblock quantization (abbreviated as B quantization): The quantization object can be either the left matrix or the right matrix. Data is blocked on all axes, and each block uses independent quantization parameters.

  - Assume the left matrix shape is (m, k). k is the reduce axis. Group data on the m and k axes by (bs, bs) blocks respectively. bs is the block size. The generated quantization parameter shape is (m/bs, k/bs).
  - Assume the right matrix shape is (k, n). k is the reduce axis. Group data on the k and n axes by (bs, bs) blocks respectively. bs is the block size. The generated quantization parameter shape is (k/bs, n/bs).
<!--
  ![Schematic Diagram](../figures/perblock量化.png)
-->
## Common Combined Quantization

- Full quantization: Generally refers to the mode of quantizing both the left and right matrices, including
  - pertensor-perchannel quantization mode (abbreviated as T-C quantization mode)
  - pertoken-perchannel quantization mode (abbreviated as K-C quantization mode)
  - pergroup-perblock quantization mode (abbreviated as G-B quantization mode)
  - pertensor-perchannel-pergroup quantization mode (abbreviated as T-CG quantization mode)
  - perblock-perblock quantization mode (abbreviated as B-B quantization mode)
- Pseudo quantization: Generally refers to the mode of quantizing the weight matrix, including perchannel quantization mode (abbreviated as C quantization mode).
- mx quantization: Essentially Microscaling quantization, which maintains model accuracy at extremely low bits (such as 1 bit) by dynamically adjusting the scaling factor. Here it refers to pergroup-pergroup quantization mode (abbreviated as G-G quantization mode), which is a special case where the quantization parameter type is FLOAT8_E8M0 and the group size is 32.
