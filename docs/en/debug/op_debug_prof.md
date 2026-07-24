# Operator Debugging and Tuning

## Debugging and Locating (AI Core Operator)

During operator execution, if operator execution failure or accuracy abnormality occurs, you can print information at each stage, such as Kernel intermediate results, for problem analysis and locating.

### 1. Host-side Log Acquisition Method

* **plog Acquisition**

   After program execution ends, you can view it in "$HOME/ascend/log" by default. The host log file storage path is as follows:

   ```sh
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   Enable environment variable ASCEND_SLOG_PRINT_TO_STDOUT to display log directly on screen (1: enable screen printing, 0: disable screen printing). Configuration example is as follows:

   ```sh
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   For log related introduction, refer to [Log Reference](https://hiascend.com/document/redirect/CannCommunitylogref). For environment variable introduction, refer to [Environment Variable Reference](https://hiascend.com/document/redirect/CannCommunityEnvRef).

* **aclnn Exception Error Information Acquisition**
   
   Use aclGetRecentErrMsg interface (refer to [acl API (C)](https://hiascend.com/document/redirect/CannCommunityCppApi)) to obtain exception information during aclnn interface invocation. Usage method is as follows:

   ```sh
   printf(aclGetRecentErrMsg());
   ```

   Print error information example is as follows:

   ```sh
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel Debugging

Common debugging methods are as follows:

* **printf**

  This interface supports printing Scalar type data, such as integers, characters, boolean, etc. For detailed introduction, refer to "Operator Debugging API > printf" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).
  
  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print current core calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping specified Tensor contents and also supports printing custom additional information, such as current line number. For detailed introduction, refer to "Operator Debugging API > DumpTensor" in [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi).
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

For problem locating in complex scenarios, such as operator hang, GM/UB access out of bounds, you can use **single-step debugging** method. For specific operations, refer to [msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug) operator debugging tool.

## Debugging and Locating (AI CPU Operator)

During operator execution, if operator execution failure or accuracy abnormality occurs, you can print information at each stage, such as Kernel intermediate results, for problem analysis and locating.

### 1. Host-side Log Acquisition Method

   Refer to AI Core operator [Host-side Log Acquisition Method](#1-host-side-log-acquisition-method)

### 2. Kernel Debugging

Common debugging methods are as follows:

* **KERNEL\_LOG Macro**

  You can use the following macro to print log information during operator execution, including DEBUG, INFO, WARN, ERROR level logs.

  ```Cpp
  KERNEL_LOG_DEBUG(fmt, …)      // fmt parameter represents format control string
  KERNEL_LOG_INFO(fmt, …)
  KERNEL_LOG_WARN(fmt, …)
  KERNEL_LOG_ERROR(fmt, …)      // Print ERROR level log by default
  ```

  To print non-ERROR level logs, you need to configure environment variable `ASCEND_GLOBAL_LOG_LEVEL` in advance. For specific usage, refer to [Environment Variable Reference](https://hiascend.com/document/redirect/CannCommunityEnvRef).

  Print example is as follows:

  ```c++
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(0);

  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    // Print error information
    KERNEL_LOG_ERROR("Invalid argument");
    return kParamInvalid;
  }

  int64_t num_elements = input0->NumElements();
  // Print input element count
  KERNEL_LOG_INFO("Num of elements is %ld", num_elements);
  ```

## Performance Tuning

### Method 1 (For Atlas A2/A3 Series Products)

During operator execution, if execution performance degradation or memory usage abnormality occurs, you can use [msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof) performance analysis tool to analyze operator execution stage metric data (such as throughput, memory usage, latency) to determine the root cause of the problem and optimize accordingly.

This chapter takes `AddExample` custom operator as an example to introduce two commonly used methods in operator tuning: operator on-board performance collection and pipeline simulation. Analyze operator Bound scenarios by collecting various pipeline metrics during operator on-board execution. Understand simulation pipeline diagrams to facilitate optimizing operator internal pipelines.

1. Prerequisites.

   After completing operator development and compilation, assuming aclnn interface method is used for invocation, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/`.

2. Collect performance data.

   When you need to collect various pipeline metrics during operator on-board execution, you can enter the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op ./test_aclnn_add_example
   ```

   The collection result is in the project `examples/add_example/examples/build/bin/OPPROF_*` directory. After collection completes, the following information is printed:

    ``` text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector             
    Task Duration(us): 97.861954 
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```

   Among them, Task Duration is the current operator Kernel latency, and Block Dim is the current operator execution core count.

   For detailed metrics of operator pipelines, refer to the `ArithmeticUtilization` file under `OPPROF_*`, which contains the proportion of current pipelines. For detailed introduction, refer to the "Performance Data File > msprof op > ArithmeticUtilization (cube and vector type instruction latency and proportion)" chapter in [msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof).

3. Collect simulation pipeline diagram.

   Before using msProf tool for operator simulation tuning, execute the following command to configure environment variables:

   ```bash
   export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH 
   ```

   Please modify the above environment variables according to the actual CANN software package installation path and AI processor model.

   Then enter the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op simulator --output=$PWD/pipeline_auto --kernel-name"AddExample" ./test_aclnn_add_example
   ```

   The collection result is in the project `$PWD/pipeline_auto/OPPROF_**` directory.
   The pipeline related file path is `OPPROF**/simulator/visualize_data.bin`, which can be viewed using [MindStudio Insight](https://www.hiascend.com/document/redirect/MindStudioInsight) tool.
   
### Method 2 (For Ascend 950PR)

During operator development, if execution performance degradation or memory usage abnormality occurs, you can use [CANN Simulator](./cann_sim.md) simulation tool to analyze operator instruction pipeline conditions to determine the root cause of the problem and optimize accordingly.

This chapter takes `AddExample` custom operator as an example to introduce the use of simulation tool. How to perform accuracy and performance tuning through simulation tool.

1. Prerequisites.

   After completing operator development and compilation, assuming aclnn interface method is used for invocation, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/`.

2. Execute simulation command to generate simulation data.

   ```sh
   cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
   ```

   The simulation result is in the project `examples/add_example/examples/build/bin/cannsim_*` directory. The pipeline related file is:

   ```sh
   trace_core0.json
   ``` 

3. Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace_core0.json) to the blank area to open. For specific parameter introduction, refer to the ["Simulation Result Analysis"](./cann_sim.md#simulation-result-analysis-instructions) chapter in CANN Simulator.
