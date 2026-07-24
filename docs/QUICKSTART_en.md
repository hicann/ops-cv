# Quick Start: Based on ops-math Repository

## Usage Notice

This guide aims to help you quickly get started with CANN and the `ops-math` operator repository. To help you quickly understand the entire operator development process, the **AddExample** operator is used as the practice object. The operator source code is located in `ops-math/examples/add_example`. The operation process is as follows:

1. **[Prerequisites](../README_en.md)**: Refer to the project README to complete environment preparation and source code download. This section will not be repeated here. For quick start scenarios, **CANNLab or Docker deployment is recommended** for simple operation.

   > **Note**: CANNLab or Docker environments provide the latest CANN package by default. If you need to experience the latest capabilities of the master branch, you can manually set up the environment.

2. **[Compile and Run](#1-compile-and-run)**: Compile the custom operator package and install it to achieve quick operator invocation.

3. **[Operator Development](#2-operator-development)**: Experience the complete loop of development, compilation, and verification by modifying the existing operator Kernel.

4. **[Operator Debugging](#3-operator-debugging)**: Master operator printing and performance collection methods.

5. **[Operator Verification](#4-operator-verification)**: Learn how to modify operator example samples to verify the functional correctness of operators under different inputs.

## 1. Compile and Run

The purpose of this stage is to **quickly experience the project standard process** and verify whether the environment can successfully perform operator source code compilation, packaging, installation, and running.

### 1. Enter Project Source Code

- CANNLab cloud development environment:

   The project source code matching the latest CANN package is provided by default. Enter the source code directory and replace ${gitCode\_id} with the developer's personal gitCode account.

   ```bash
   cd /mnt/workspace/gitCode/${gitCode_id}/ops-math
   ```

- Non-CANNLab cloud development environment:

  According to the source code and CANN version compatibility relationship in the [release repository](https://gitcode.com/cann/release-management), execute the following command to download the source code. Replace ${tag\_version} with the target branch tag, for example, 9.0.0.

  ```bash
  git clone -b ${tag_version} https://gitcode.com/cann/ops-math.git && cd ops-math
  ```

> Note: If you need to switch the source code branch version, refer to the following instructions.
>
> 1. Execute `git branch` in the source code directory to query the current source code version.
> 2. Execute `git checkout ${tag_version}` in the source code directory to switch to the target branch source code. Note that the source code and CANN version compatibility relationship must be satisfied. If the source code already exists, execute `git pull` to pull the latest source code.

### 2. Compile AddExample Operator

This guide defaults to **single operator compilation**: only build the target operator, with short compilation time, suitable for quick start and daily development. General command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

> If you need to compile the entire operator library (omit `--ops`), refer to [Source Code Build Guide · Full Compilation (ops-math package)](zh/install/compile.md#ops-math包).
> **Note**: Before compiling, ensure that the CANN environment variables are configured. Otherwise, compilation may fail due to not finding `ASCEND_HOME_PATH`. For default path installation, execute:
>
> ```bash
> source /usr/local/Ascend/cann/set_env.sh
> ```

Taking the AddExample operator as an example, the compilation command is as follows:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
```

The ${soc_version} values corresponding to product names are as follows. Pass parameters according to the actual scenario.

- Atlas A2 training series products/Atlas A2 inference series products: The value is ascend910b
- Atlas A3 training series products/Atlas A3 inference series products: The value is ascend910_93
- 950 series products: The value is ascend950

If the following information is displayed, the compilation is successful.

```bash
Self-extractable archive "cann-ops-math-custom_linux-${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install AddExample Operator Package

```bash
./build_out/cann-ops-math-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path, where ```${ASCEND_HOME_PATH}``` represents the CANN software installation directory.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variables to ensure it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_math/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run Operator Sample

General run command format: `bash build.sh --run_example <operator name> <run mode> <package mode>`.

Taking AddExample as an example, it provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

Expected output: Print the addition calculation result of the `AddExample` operator, indicating that the operator has been successfully deployed and executed correctly. The sample input is fixedly generated by code (`selfX` is an increasing sequence, `selfY` is a random number). The output format is as follows:

add_example first input[0] is: 1.000000, second input[0] is: (random value), result[0] is: (Add result)
add_example first input[1] is: 2.000000, second input[1] is: (random value), result[1] is: (Add result)
...

## 2. Operator Development

The purpose of this stage is to attempt **modifying the kernel function code** of the successfully running AddExample operator.

### 1. Modify Kernel Implementation

Find the core kernel implementation file of the AddExample operator `ops-math/examples/add_example/op_kernel/add_example.h`, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, currentNum);
    AscendC::Mul(zLocal, xLocal, yLocal, currentNum);
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
    bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
    ```

    > **Note**: Fill in `${soc_version}` according to the actual chip model. The value method is the same as described in [Compile AddExample Operator](#2-compile-addexample-operator).

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-math-*linux*.run
    ```

3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

4. **Success indicator**: The output result becomes a multiplication result (`result = first input × second input`). The output format is as follows:

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: (random value), result[0] is: (Mul result)
    add_example first input[1] is: 2.000000, second input[1] is: (random value), result[1] is: (Mul result)
    ...
    ```

## 3. Operator Debugging

This stage takes AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and positioning.

### 1. Printing

If the operator has execution failure, accuracy abnormality, or other problems, add printing for problem analysis and positioning.

Please modify the code in `examples/add_example/op_kernel/add_example.h`.

* **printf**

  This interface supports printing Scalar type data, such as integers, character type, and boolean type. For detailed introduction, refer to "Operator Debugging API > printf" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).

  ```c++
  int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
  blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
  ubLength_ = tilingData->ubFactor;
  // Print the current kernel calculation Block length
  AscendC::PRINTF("Tiling blockLength is %lld\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of a specified Tensor and also supports printing custom additional information, such as the current line number. For detailed introduction, refer to "Operator Debugging API > DumpTensor" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).

  ```c++
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Collection

After the operator function verification is correct, you can collect operator-level performance data through the `msprof op` command.

- **Generate executable file**

    Call the example sample of the AddExample operator to generate an executable file (test_aclnn_add_example), which is located in the project `ops-math/build` directory.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

- **Collect performance data**

    Enter the AddExample operator executable file directory `ops-math/build/` and execute the following command:

    ```bash
    msprof op --application="./test_aclnn_add_example"
    ```

    After execution, the operator basic information (such as Op Name, Op Type, Task Duration, Block Dim) and performance bottleneck prompts will be printed directly.

The collection results are saved in the `OPPROF_*` folder under the project `ops-math/build/` directory. After the command execution is completed, the performance data file will be automatically parsed and exported. For further interpretation of various performance indicators (such as pipeline proportion and bandwidth utilization), refer to the [msProf Tool Usage Guide](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof).

## 4. Operator Verification

This stage verifies the functional correctness of the operator in multiple scenarios by modifying the input data in the AddExample operator example sample.

### 1. Modify Test Input

Find and edit `ops-math/examples/add_example/examples/test_aclnn_add_example.cpp` for `AddExample`, and modify the shape and values of the input tensors.

**Modify input/output data**: Modify the shape information of input and output, as well as initialize data, and construct corresponding input and output tensors.

```c++
int main() {
    // ...initialization code ...

    // === ① Modify selfX input ===
    // Before modification: shape = {32, 4, 4, 4}, values are an increasing sequence starting from 1.0 with a step of 1
    // After modification: Change input shape to {8, 8, 8, 8} and fill with different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // You can use a loop to fill more distinguishable data, such as an increasing sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill with cyclic values of 0-9
    }
    // === ② Refer to selfX, similarly modify the shape and data of selfY (second input) and out (output) ===

    // ...subsequent execution code ...
}
```

> **Note**: When modifying the shape, you need to synchronously modify the length of the corresponding host-side data vector (in this example, the vector lengths of `selfX`, `selfY`, and `out` all need to be changed from the default 2048 to 4096 = 8×8×8×8), and ensure that the three shapes are consistent. Otherwise, data out-of-bounds or result mismatch will occur.

### 2. Recompile and Verify

1. Since only the example test code was modified, there is no need to recompile the operator package.

2. Re-execute the verification command:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

3. Observe whether the operator output result meets expectations.

## Conclusion

After experiencing the above process, you have basically completed the operator development process. If you want to further contribute new operators or learn more advanced development, debugging, and other skills, visit this project's README to learn [Advanced Tutorials](../README_en.md#learning-tutorials) and [Contribution Guide](../README_en.md#related-information).
