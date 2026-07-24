# Cross-Platform Migration Guide

This guide introduces the adaptation points and solutions for operator migration across multiple platforms. Taking the operator migration from Atlas A2 series to Ascend 950 series as an example, it compares hardware architecture differences and related adaptation points, and provides relevant operator adaptation samples.

## 1. Hardware Architecture and Specification Parameter Comparison
<!--
### Atlas A2 Series Hardware Architecture

<div align="center">
  <img src="../figures/Atlas A2硬件架构.png" width="900" alt="Atlas A2 Hardware Architecture" />
</div>

### Ascend 950 Series Hardware Architecture

<div align="center">
  <img src="../figures/Ascend 950硬件架构.png" width="900" alt="Ascend 950 Hardware Architecture" />
</div>
-->
### Generation Specification Parameter Comparison

Usually multiple product models are divided according to different application scenarios, processes or hardware configurations. Each model may have certain differences in performance, resource configuration and other aspects. For ease of explanation and direct comparison, this section selects representative configurations as parameter display and difference analysis objects. Other related adjustments should be based on actual manuals or official releases.

<table>
  <tr>
    <th colspan="2" style="width: 25%;">Specification Item</th>
    <th style="width:37.5%;">Atlas A2</th>
    <th style="width:37.5%;">Ascend 950</th>
  </tr>
  <tr>
    <td rowspan="4">AICore</td>
    <td>Core Count</td>
    <td>24</td>
    <td>32</td>
  </tr>
  <tr>
    <td>Frequency</td>
    <td>1.8</td>
    <td>1.65</td>
  </tr>
  <tr>
    <td>Cube Compute Specification</td>
    <td>353T/376T @BF16,FP16</td>
    <td>426T@BF16,FP16 757T@FP8,HIFP8,MXFP8,INT8 1514T@MXFP4</td>
  </tr>
  <tr>
    <td>Vector Compute Specification (FP16)</td>
    <td>23.5T</td>
    <td>54T</td>
  </tr>
  <tr>
    <td rowspan="2">Memory</td>
    <td>Memory Capacity (GB)</td>
    <td>64</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Memory Bandwidth</td>
    <td>1.6TB/s</td>
    <td>1.6TB/s</td>
  </tr>
</table>

## 2. Hardware Capability Changes Introduced Adaptation Points

<table>
  <tr>
    <th style="width: 25%;">Hardware Unit</th>
    <th style="width:35%;">Hardware Capability Change</th>
    <th style="width:40%;">Typical Impact Scope</th>
  </tr>
  <tr>
    <td rowspan="5">Transfer Unit</td>
    <td>Removed L1 to GM data path</td>
    <td>Kernels relying on L1 direct write-back to GM need to change to L1→UB→GM or L0C/FIXPIPE→GM path; related DataCopy links, event synchronization and buffer planning need adjustment</td>
  </tr>
  <tr>
    <td>Removed GM to L0A, L0B data path</td>
    <td>GM→L0A/L0B direct connection is no longer available, need to complete through GM→L1→L0A/L0B; L1 block strategy and MTE1/2 pipeline need reconstruction</td>
  </tr>
  <tr>
    <td>ND DMA flexible data transfer, supports on-the-fly ND->NZ conversion</td>
    <td>Can use ND2NZ/DN2NZ to complete format conversion in MTE2 phase, reducing intermediate buffer and format conversion overhead; need to pay attention to stride, alignment and NZ shape mapping</td>
  </tr>
  <tr>
    <td>Supports Cube->Vector efficient internal data path: L1->UB, L0C->UB, FIXP->UB</td>
    <td>Can perform intermediate accumulation/activation/fusion on UB side (such as K-cut accumulation, post-processing), reducing GM round trips; corresponding synchronization and pipeline partitioning need adjustment</td>
  </tr>
  <tr>
    <td>Introduced collective communication accelerator CCU1.0</td>
    <td>Communication-computation fusion operators adjust HcclServerType in Eager mode; use CCU series GE interfaces in Graph mode</td>
  </tr>
  <tr>
    <td rowspan="3">Compute Unit</td>
    <td>Vector added Regbase paradigm</td>
    <td>Original Membase-based access patterns, alignment methods, register count assumptions need re-examination; templates/tiling may need update to Regbase version</td>
  </tr>
  <tr>
    <td>Cube no longer supports int4_t</td>
    <td>All operators using int4_t need to switch to supported data types (such as int8), and update quantization solution logic</td>
  </tr>
  <tr>
    <td>Does not support 4:2 sparse matrix computation</td>
    <td>Kernels originally relying on 4:2 sparse feature for speedup need to change to dense or other supported sparse strategies, and update performance expectation documentation</td>
  </tr>
  <tr>
    <td rowspan="1">Storage Unit</td>
    <td>Local Buffer memory improvement: Cube L0C 256KB, Vector UB 256KB</td>
    <td>Larger L0C/UB allows increasing basic block and double buffer capacity, reducing K-cut/block-cut rounds; need to re-evaluate L1/L0/UB ratio and tile size</td>
  </tr>
  <tr>
    <td rowspan="2">Others</td>
    <td>Multi-core simultaneous access to Global Memory same address performance optimization</td>
    <td>Templates involving matrix multiplication related operators can be optimized</td>
  </tr>
  <tr>
    <td>SIMT</td>
    <td>After SIMT introduction, thread-level parallelism can be used to handle branches/irregular computation, but needs adaptation for thread partitioning, shared memory and synchronization semantics; some Vector implementations can be migrated to SIMT version</td>
  </tr>
</table>

## 3. Recommended Migration Steps

1. Confirm whether the compute units (Cube/Vector) involved in the operator and the data types supported by corresponding units have differences across platforms.
2. Confirm whether the data transfer units involved (ND->NZ, GM->Lx, collective communication, etc.) have differences across platforms.
3. Modify item by item according to hardware capability change points (Vector architecture, Cube supported data types, L1/L0/UB size, CCU communication, etc.).
4. Refer to operator migration samples to adjust/complete Atlas A2/Ascend 950 branch logic.

## 4. Operator Migration Samples

### Cube Matrix Computation Operators

#### Global Memory Same Address Access Conflict Optimization

Ascend 950 hardware adds same address request parallel processing feature, no need to additionally avoid same address access conflicts in various core partitioning scenarios. During migration, the core partitioning strategy designed for "staggered conflict avoidance" on Atlas A2 can be simplified to more regular sliding window templates (such as row group window + column direction round-trip scanning), reducing invalid offsets and redundant address transformations. In practice, it is recommended to first aim for functional equivalence and retain original tile size, then gradually relax core partitioning constraints, combine profiling data to observe key indicators such as MAC utilization, MTE2 utilization, L2 hit rate, and confirm whether template adjustment brings stable benefits.
<!--
<div align="center">
  <img src="../figures/SWAT滑动窗口模板.png" width="900" alt="SWAT Sliding Window Template" />
</div>
-->
#### Tile Size Adjustment

On Atlas A2, L0C size is 128KB. Ascend 950 increases to 256KB, meaning single pass can carry larger accumulation result blocks. During migration, priority can be given to increasing Tile block partitioning granularity or increasing K-direction single-round processing depth to reduce block-cut and K-cut rounds, lowering loop control and transfer overhead. At the same time, need to re-balance L1/L0/UB capacity budget to avoid L0C enlargement squeezing A/B/scale buffer causing pipeline breakpoints.

### Vector Vector Computation Operators

#### SIMT

Ascend 950 series adds SIMT unit. SIMT has significant advantages over SIMD in handling non-regular discrete access, suitable for scenarios with discontinuous addresses, large access span variations, and inconsistent branch paths (such as scatter/gather, index rearrangement, sparse update, etc.).

During migration, it is recommended to first identify "access-dominated" and "low vectorization efficiency" operator sub-processes: if the original SIMD implementation has many mask branches, high invalid lane ratio, or requires complex address assembly, that part can be rewritten as SIMT path, usually reducing control overhead and increasing effective access throughput.

In practice, need to focus on the following: first, thread task partitioning should match data sparsity to avoid extremely unbalanced thread load; second, reduce pipeline stalls caused by high-frequency random access, try to complete index regularization and bucketing upstream; third, decouple boundary processing from main path to avoid introducing too many branches in hot loops. It is recommended to compare "pure SIMD implementation" and "SIMD+SIMT hybrid implementation" after migration, and select optimal strategy based on data distribution rather than fixing on a single path.

**Taking gather_v2 operator as an example: SIMD vs SIMT implementation comparison**

The gather_v2 operator performs gather based on the combined tail axis as unit, so the template selection basis is: when tail axis ≤ 2048, use SIMT template; when tail axis > 2048, use SIMD template. Because when tail axis is small, discrete access to multiple discontinuous small block addresses is needed, SIMT is more efficient. The following compares the core differences between two implementations:

**1. Programming Model Difference**

SIMD implementation adopts traditional vectorization programming model, requiring explicit UB buffer and pipeline queue management:

```Cpp
// SIMD: Use queue mechanism to manage data buffer
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
TBuf<QuePosition::VECCALC> indexBuf_;

// SIMD: Process row by row, explicit transfer and synchronization
for (int64_t j = 0; j < rows; j++) {
    INDICES_T index = GetIndex(yIdx, indiceEndIdx);  // Scalar read index
    int64_t xIndex = index * tilingData_->innerSize;
    DataCopyPad(xLocal[j * colsAlign], xGm[offset], dataCoptExtParams, dataCopyPadExtParams); // Batch continuous data transfer in
}
inQueue_.EnQue<int8_t>(xLocal);  // Enqueue waiting for output
```

SIMT adopts thread-level parallel model, each thread independently processes elements:

```Cpp
// SIMT: Use thread-level parallelism, no explicit buffer management needed
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
    for (INDEX_SIZE_T index = Simt::GetThreadIdx(); 
         index < currentCoreElements;
         index += Simt::GetThreadNum()) {  // Thread jumping parallelism
        // Each thread independently calculates single-point index and accesses
        INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
        INDICES_T indicesValue = indices[gatherI];  // Directly access GM based on single-point index gatherI
        y[yIndex] = idxOutOfBound ? 0 : x[xIndex];  // Directly write back to GM
    }
}
```

**2. Access Pattern Difference**

| Feature | SIMD Implementation | SIMT Implementation |
|---------|---------------------|---------------------|
| Data Access | Explicit transfer to UB through DataCopyPad | Thread directly accesses GM through `__gm__` pointer |
| Buffer Management | Need AllocTensor/EnQue/DeQue/FreeTensor | No explicit buffer needed, hardware automatically manages |
| Synchronization Mechanism | Explicit event synchronization (HardEvent::MTE2_V, etc.) | Implicit synchronization between threads |

**3. Applicable Scenario Difference**

SIMD is suitable for scenarios with continuous access to large block addresses, efficiently processing continuous data through vectorization instructions;

SIMT is suitable for discrete access, with threads processing in parallel.

#### Regbase

Ascend 950 series introduces Regbase programming paradigm. Compared to traditional Membase (Vector API) programming, Regbase is closer to underlying hardware register operations, providing finer vectorization control capabilities.

**Features**

- Uses low-level APIs under `AscendC::MicroAPI` namespace
- Directly operates registers `RegTensor<T>` instead of explicitly managing UB buffer queues
- Implements flexible element-level mask control through `MaskReg`

**Comparison with Membase Programming Model**

| Feature | Membase (Traditional Vector API) | Regbase (MicroAPI) |
|---------|----------------------------------|---------------------|
| Data Carrier | `LocalTensor<T>` + Queue mechanism | `RegTensor<T>` register |
| Memory Management | Explicit Alloc/EnQue/DeQue/Free | Automatic register allocation |
| Mask Control | Function parameter control | `MaskReg` register control |
| Data Transfer | `DataCopy`/`DataCopyPad` | `MicroAPI::DataCopy` + distribution mode |

**Code Examples**

```cpp
__simd_vf__ __aicore__ void GenIndexBuf(ubuf int32_t* helpAddr, int32_t colFactor)
{
    // Declare register tensors
    AscendC::MicroAPI::RegTensor<int32_t> v0;
    AscendC::MicroAPI::RegTensor<int32_t> v1;
    AscendC::MicroAPI::RegTensor<int32_t> vd1;
    
    // Create full mask
    AscendC::MicroAPI::MaskReg preg = 
        AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    
    // Scalar copy to register
    AscendC::MicroAPI::Duplicate(v1, colFactor, preg);
    // Generate sequence [0, 1, 2, ...]
    AscendC::MicroAPI::Arange(v0, 0);
    // Vector operations
    AscendC::MicroAPI::Div(vd1, v0, v1, preg);
    AscendC::MicroAPI::Mul(vd2, vd1, v1, preg);
    AscendC::MicroAPI::Sub(vd3, v0, vd2, preg);
    // Register data write back to UB
    AscendC::MicroAPI::DataCopy(helpAddr, vd3, preg);
}
```

```cpp
// Dynamic mask: handling incomplete tail data
__simd_vf__ __aicore__ void GatherProcess(ubuf int8_t* curYAddr, uint16_t repeatimes, uint16_t computeSize)
{
    MicroAPI::RegTensor<int8_t> vregTemp;
    MicroAPI::MaskReg preg;
    
    for (uint16_t r = 0; r < repeatTimes; r++) {
        // Update mask based on remaining element count
        preg = MicroAPI::UpdateMask<int8_t>(sreg);
        // Create address offset register
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
        MicroAPI::DataCopy(vregTemp, curXAddr, offset);
        // Masked data store
        MicroAPI::DataCopy(curYAddr, vregTemp, offset, preg);
    }
}
```

```cpp
// Data aggregation
__VEC_SCOPE_
{
    MicroAPI::RegTensor<uint32_t> indicesReg;
    MicroAPI::RegTensor<int32_t> vd0;
    
    for (uint16_t indices = 0; indices < indicesLoopNum; indices++) {
        // Load index (E2B distribution mode: broadcast scalar to vector)
        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_E2B_B32>(indicesReg, indicesAddr);
        // Gather data aggregation based on index
        MicroAPI::DataCopyGather(vd0, curXAddr, indicesReg, preg);
        // Data block copy output
        MicroAPI::DataCopy<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            curYAddr, vd0, blockStride, preg);
    }
}
```

**Key Regbase API Description**

| API Category | API Name | Function Description |
|--------------|----------|----------------------|
| Register Type | `RegTensor<T>` | Vector register tensor type |
| Mask Type | `MaskReg` | Mask register type |
| Mask Creation | `CreateMask<T, Pattern>()` | Create mask (ALL/HALF, etc. modes) |
| Mask Update | `UpdateMask<T>(count)` | Dynamically update mask based on remaining element count |
| Scalar Operation | `Duplicate(reg, val, mask)` | Copy scalar value to all register elements |
| Sequence Generation | `Arange(reg, start)` | Generate consecutive sequence |
| Arithmetic Operations | `Add/Sub/Mul/Div(dst, src1, src2, mask)` | Vector arithmetic operations |
| Scalar Operations | `Adds/Muls(dst, src, scalar, mask)` | Vector and scalar operations |
| Type Conversion | `Cast<DT, ST>(dst, src, mask)` | Data type conversion |
| Comparison Operations | `Compare<T, CMPMODE>(mask, src1, src2, pred)` | Vector comparison generating mask |
| Data Load | `DataCopy<T, LoadDist>(reg, addr)` | Load from UB to register |
| Data Store | `DataCopy<T>(addr, reg, mask)` | Store from register to UB |
| Gather | `DataCopyGather(dst, base, indices, mask)` | Collect data based on index |
| Address Offset | `CreateAddrReg<T>(loop, stride)` | Create loop address offset register |

**Distribution Mode (LoadDist) Description**

| Mode | Description | Typical Use |
|------|-------------|-------------|
| `DIST_NORM` | Normal continuous load | Continuous data processing |
| `DIST_UNPACK_B16` | 16-bit unpack load | FP16/BF16 to FP32 |
| `DIST_BRC_B32/B16` | Broadcast load | Scalar scale broadcast |
| `DIST_E2B_B32` | Scalar to vector broadcast | Index value broadcast |

**Migration Suggestions**

1. Scenarios suitable for Regbase: Need fine control of register allocation, complex mask logic, Gather/Scatter access patterns
2. Scenarios to retain Membase: Simple continuous data transfer and computation, double buffer pipeline
3. Hybrid use: Can combine both paradigms in the same operator, use Regbase for core computation logic, use Membase for data transfer management

### Cube-Vector Fusion Operators

#### MTE Data Transfer Path Changes

Ascend 950 new architecture introduces direct connection paths between UB2L1 & L0C2UB, enabling fast transfer of matrix computation data, aiming to simplify CV fusion operator development and improve performance.
<!--
<div align="center">
  <img src="../figures/Ascend950新增CV直连通路.png" width="700" alt="Ascend950 New CV Direct Connection Path" />
</div>
-->
**Matrix Transfer In**

Enable UB to L1 (UB2L1) direct connection path, through DataCopy interface, supports vector computation results of fusion operators directly transferred to L1.

**Matrix Transfer Out**

Enable L0C to UB (L0C2UB) direct connection path, through DataCopy interface, supports matrix computation results of fusion operators directly transferred to UB for subsequent vector computation.

For K-cut or multi-stage fusion scenarios, can change "L0C write back to GM then read back to UB" to "L0C direct to UB accumulation/post-processing", reducing GM round-trip bandwidth pressure and latency. During migration, it is recommended to complete intermediate result merging, activation/quantization pre-processing on UB side, and explicitly organize event synchronization order of MTE1/MTE2/MTE3 and compute units to ensure cross-unit pipeline continuity, avoiding data visibility or synchronization timing issues introduced by new paths. Key enabling interface definitions can be referenced:

```cpp
// 1. New: Transfer in interface adds UB2L1 Nd2Nz transfer, supports both Src&Dst as LocalTensor form
template <typename T>   
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const Nd2NzParams& intriParams);

// 2. New: Transfer out interface adds L0C2UB transfer, supports direct transfer from L0C to UB, supports both Src&Dst as LocalTensor form
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams);
template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 {
    // ... 
    uint8_t dualDstCtl = 0;
};

// 3. Capability enhancement: Cross-core synchronization interface adds mode 3
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

#### Cross-Core Synchronization Semaphore Matching

`CrossCoreSetFlag` and `CrossCoreWaitFlag` are cross-core synchronization semaphore interfaces, widely used for data dependency and coordination control between multiple cores. Essentially, they implement decoupling and orderly progression of data processing stages between different AICores in a "semaphore" manner, commonly used for pipeline control, double buffer switching, cross-core collaboration and other scenarios.

- `CrossCoreSetFlag`: After the current core (or thread) completes data processing of a certain stage, it actively sets the specified flag signal, informing the dependent party (usually other cores or downstream pipeline stages) that this stage is complete and can continue executing subsequent processes.
- `CrossCoreWaitFlag`: The current core (or thread) needs to wait for a certain flag signal to be set (that is, the dependent data or event is complete), and will only continue executing downward after detecting the flag.

The essence of this set of semaphores is to ensure consistent synchronization sequences between multiple threads/multiple pipeline stages, preventing data races or deadlocks and other hardware exceptions due to resources not ready or dependencies not completed. For detailed interface descriptions, please refer to official documentation: [CrossCoreSetFlag and CrossCoreWaitFlag Cross-Core Synchronization Interface Details](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0273.html).

On Ascend 950, `CrossCoreWaitFlag` and `CrossCoreSetFlag` counts must match strictly, and it is recommended to design in pairs in the order of "produce first then consume" within the same synchronization semantic domain. On Atlas A2, if there are extra `CrossCoreSetFlag` semaphores between operators, HWTS will perform special processing to clear the counter. Ascend 950 series, to reduce hardware overhead, no longer relies on such fallback mechanisms, requiring one-to-one matching of cross-core synchronization semaphores within single operator kernels, otherwise inevitable deadlock will occur.

During migration, please focus on checking the following issues: first, exception branches returning early causing only `Set` execution without corresponding `Wait` (or vice versa); second, multi-stage pipelines reusing the same `flagId` but with overlapping lifecycles, causing "cross-stage crosstalk"; third, condition-triggered synchronization inside loops but loop boundaries not aligned, causing iteration count inconsistency. The above issues may be masked on Atlas A2, but will directly manifest as blocking timeout or deadlock on Ascend 950. For operators with complex cross-core pipelines, can first construct minimal dataset for single-stage verification, then gradually add double buffer and multi-stage to reduce the complexity of locating synchronization issues.

### Collective Communication Operators

Ascend 950 introduces collective communication accelerator CCU1.0, reducing access requirements and scheduling latency. To effectively utilize this feature, operator cross-chip communication method is changed from A2's AICPU to CCU communication.

**Eager Mode**

In the second segment interface of aclnn two-segment interface, specify collective communication type for operator executor aclOpExecutor.

Taking [MatmulAllReduce](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce) operator migration adaptation as an example:
Set NnopbaseSetHcclServerType enumeration value, A2 is NNOPBASE_HCCL_SERVER_AICPU, 950 is NNOPBASE_HCCL_SERVER_TYPE_CCU.

```CPP
// ...
aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // ...
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    // ...
    return ACLNN_SUCCESS;
}
```

**Graph Mode**

1. Used for resource calculation and allocation, in CalcParamFunc callback interface involving auxiliary stream related information, distinguish auxiliary stream collective communication type for GE context.
2. Used for setting custom tasks on main stream/auxiliary stream and parameter customization, in GenerateTask callback interface, distinguish two sets of GE KernelLaunch interfaces, separately call AICPU communication or CCU communication creation and customization processes.

Static graph GE side creates communication task type, A2 is aicpu kfc server + kfc_stream; 950 is ccu server + ccu_stream. Related code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
ge::Status MatmulAllReduceCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    // A2
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}
// ...
```

Static graph GenTask invocation interfaces differ, processes vary. Related code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
// A2
ge::Status MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>>& tasks) {
    // ...
    // aicpu task
    ge::KernelLaunchInfo aicpu_task =
        ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
    // ...
}


// 950
ge::Status Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    // ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    // ...
}

ge::Status MatmulAllReduceGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    // A2
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}
// ...
```

## 5. Common Issues and Performance Tuning Suggestions (FAQ/Performance Tips)

If operator performance on Ascend 950 decreases instead of increases, prioritize checking:

1. Whether still using Atlas A2's staggered core partitioning template
2. Whether CCU communication is not enabled and still using AICPU
3. Whether tiling still uses Atlas A2's L1/L0/UB partitioning strategy, causing Ascend 950's larger on-chip cache to not be fully utilized
