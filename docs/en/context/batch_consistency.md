# Batch Consistency

## Introduction

During application development, some operators pursue higher performance and may produce inconsistent computation results for the same token across different batch sizes or different positions within the same batch.

Currently, for some operators under the same runtime environment, you can configure the computation process to adopt the Batch Consistency algorithm so that the computation results remain completely identical regardless of how the inputs are combined.

**Batch Consistency algorithm**: For a given token, the output must be bit-for-bit identical regardless of its position within the batch, the batch size, or which other tokens it is batched with.

## Notes

- Generally, it is recommended not to enable Batch Consistency computation, because enabling it for an operator causes some performance degradation for related operators, which may reduce the single-run performance of the model. However, in scenarios such as experiments, debugging, and regression testing that require identical results across multiple runs to locate problems or experiment with algorithms, Batch Consistency computation can improve efficiency.

- The current configuration is a process-level switch.

- **Version constraint**: TorchNPU version 26.2.0 or later, and CANN version 9.2.0 or later.

## Usage

Currently, the mainstream invocation methods for CANN operators are the aclnn API or the PyTorch API (torch_extension). Some operator APIs implement Batch Consistency by default, while others implement non-Batch-Consistency by default. For operators that do not implement Batch Consistency by default, some can be manually configured for Batch Consistency.

- **Calling the aclnn API**

  In this scenario, enable Batch Consistency through the "Runtime Configuration > aclrtSetSysParamOpt" interface (process-level) in the [Runtime API](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=3` to enable Batch Consistency computation.

- **Calling the PyTorch API**

  In this scenario, enable Batch Consistency computation through the `torch_npu.npu.set_deterministic_level` interface in the [TorchNPU Custom API](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/overview.md).

For operator APIs of different frameworks, refer to the specific aclnn API or PyTorch API documentation for their default Batch Consistency implementation strategy.
