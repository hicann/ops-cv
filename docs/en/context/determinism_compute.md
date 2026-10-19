# Deterministic Computation

## Introduction

During application development, due to differences in CANN versions or NPU models, the results of running the same operator multiple times may not be completely consistent; even with the same random seed under the same conditions, multiple NPU executions may produce different results. This difference usually stems from asynchronous multi-threaded execution in the operator implementation, which changes the accumulation order of floating-point numbers.

In some special scenarios (such as experiments, debugging, and regression testing), the operator needs to produce the same output across multiple runs to locate problems or perform algorithm analysis. This approach is usually called "Deterministic Computation".

## Notes

- Generally, it is recommended not to enable Deterministic Computation, because an operator using Deterministic Computation usually executes more slowly than non-deterministic computation, which may reduce the single-run performance of the model. However, in scenarios such as experiments, debugging, and regression testing that require consistent results across multiple runs to locate problems or analyze algorithms, Deterministic Computation can improve efficiency.

- Thread note: The deterministic state can be set only once in the same thread; if set multiple times, the last valid setting prevails. A valid setting means that after the deterministic state is set, an operator task is actually dispatched. If only set but not dispatched, the deterministic variable is enabled but not delivered to the operator, and the operator will not execute. Setting the deterministic state multiple times in one thread is not recommended for now. This issue exists whether the binary is enabled or disabled and will be resolved in a later version.

## Usage

Currently, the mainstream invocation methods for CANN operators are the aclnn API or the PyTorch API (torch_extension). Some operator APIs implement Deterministic Computation by default, while others implement non-deterministic computation by default. For operators that do not implement Deterministic Computation by default, some can be manually configured for Deterministic Computation.

- **Calling the aclnn API**

  In this scenario, enable Deterministic Computation through the "Runtime Configuration > aclrtSetSysParamOpt" interface (process-level) in the [Runtime API](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=1` to enable Deterministic Computation.

- **Calling the PyTorch API**

  In this scenario, enable Deterministic Computation through the native PyTorch switch (`torch.use_deterministic_algorithms`).

For operator APIs of different frameworks, refer to the specific aclnn API or PyTorch API documentation for their default Deterministic Computation implementation strategy.
