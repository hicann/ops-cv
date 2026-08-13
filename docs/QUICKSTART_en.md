# Quick Start: Based on ops-cv Repository

## Prerequisites

This guide aims to help you quickly get started with CANN and the `ops-cv` operator repository. To facilitate quick understanding of the entire operator development process, the **AddExample** operator will be used as the practice object. Its source files are located in `ops-cv/examples/add_example`. The specific operation process is as follows:

1. **[Environment Deployment](zh/install/quick_install.md)**: Complete software installation and source code download. This will not be described in detail here. For quick start scenarios, **WebIDE or Docker environment is recommended**, with simple installation operations.
    
    > **Note**: WebIDE or Docker environment provides the latest commercial release CANN package by default. If you need to experience the latest capabilities of the master branch, you can manually set up the environment. Note whether the software package matches the source code version.
   
2. **[Compile and Run](#1-compile-and-run)**: Compile the custom operator package and install it to achieve quick operator invocation.

3. **[Operator Development](#2-operator-development)**: Experience the complete closed loop of development, compilation, and verification by modifying the existing operator Kernel.

4. **[Operator Debugging](#3-operator-debugging)**: Master operator printing and performance collection methods.

5. **[Operator Verification](#4-operator-verification)**: Learn how to modify operator example samples to verify operator functional correctness under different inputs.

## 1. Compile and Run

The purpose of this stage is to **quickly experience the project standard process** and verify whether the environment can successfully perform operator source code compilation, packaging, installation, and running.

### 1. Enter the Project Directory
    
After the environment is ready (note that the software matches the source code version), enter the project directory.
    
- For Docker deployment or manual installation scenarios, the project source code is located at

```bash
cd ops-cv
```

- For WebIDE scenarios, the project source code is located at

```bash
cd /mnt/workspace/ops-cv
```

### 2. Compile AddExample Operator

Compile the specified operator. The general compilation command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

Taking the AddExample operator as an example, the compilation command is as follows:

```bash
bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
```

If the following information is displayed, the compilation is successful.

```bash
Self-extractable archive "cann-ops-cv-custom-linux.${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install AddExample Operator Package

```bash
./build_out/cann-ops-cv-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path. ```${ASCEND_HOME_PATH}``` represents the CANN software installation directory.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variables to ensure it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_cv/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run Operator Sample

The general running command format: `bash build.sh --run_example <operator name> <running mode> <package mode>`.

Taking AddExample as an example, it provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

Expected output: Print the addition calculation result of the `AddExample` operator, indicating that the operator has been successfully deployed and executed correctly.

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## 2. Operator Development

The purpose of this stage is to attempt to **modify the kernel function code** of the successfully running AddExample operator.

### 1. Modify Kernel Implementation

Find the core kernel implementation file of the AddExample operator `ops-cv/examples/add_example/op_kernel/add_example.h`, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Compile and Verify

Repeat the steps in the [Compile and Run](#1-compile-and-run) section:

1. **Recompile**:

    First return to the project root directory. The compilation command is as follows:

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
    ```

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-cv-*linux*.run
    ```
    
3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

4. **Success Sign**: The output result becomes the multiplication result.

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## 3. Operator Debugging

This stage takes AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and positioning.

### 1. Printing

If the operator has execution failure, accuracy abnormality, or other problems, add printing for problem analysis and positioning.

Please modify the code in `examples/add_example/op_kernel/add_example.h`.

* **printf**

  This interface supports printing Scalar type data, such as integers, character type, Boolean type, etc. For detailed introduction, refer to "Operator Debugging API > printf" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).
  
  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // Print the current kernel calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of a specified Tensor. It also supports printing custom additional information, such as the current line number. For detailed introduction, refer to "Operator Debugging API > DumpTensor" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Collection

After the operator function is verified correctly, you can collect operator performance data through the `msprof` tool.

 - **Generate Executable File**
   
    Invoke the AddExample operator example sample to generate an executable file (test_aclnn_add_example). This file is located in the project `ops-cv/build` directory.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

 - **Collect Performance Data**

    Enter the AddExample operator executable file directory `ops-cv/build/` and execute the following command:

    ```bash
    msprof --application="./test_aclnn_add_example"
    ```

The collection result is in the project `ops-cv/build/` directory. After the msprof command is executed, it will automatically parse and export the performance data result file. For detailed content, refer to [msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251).

## 4. Operator Verification

This stage verifies the functional correctness of the operator in multiple scenarios by modifying the input data in the AddExample operator example sample.

### 1. Modify Test Input

Find and edit the `ops-cv/examples/add_example/examples/test_aclnn_add_example.cpp` of `AddExample`, and modify the shape and values of the input tensor.

**Modify Input/Output Data**: Modify the shape information of input and output, as well as initialize data, and construct corresponding input and output tensors.

```c++
int main() {
    // ... Initialization code ...
    
    // === ① Modify selfX input ===
    // Before modification: shape = {32, 4, 4, 4}, values all 1
    // After modification: Change input shape to {8, 8, 8, 8}, and fill with different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 * 8
    // You can use a loop to fill more distinguishable data, such as an increasing sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill with cyclic values 0-9
    }
    // === ② Refer to selfX, similarly modify selfY and selfZ inputs ===
    
    // ... Subsequent execution code ...
}
```

### 2. Recompile and Verify

1. Since only the example test code was modified, there is no need to recompile the operator package.

2. Re-execute the verification command:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

3. Observe whether the operator output result meets expectations.

## Conclusion

After experiencing the above process, you have basically completed an operator development. If you want to contribute operators or learn more advanced skills, please visit this project README to further understand the [Learning Tutorials](../README_en.md#learning-tutorials) and [Contribution Guide](../README_en.md#related-information).
