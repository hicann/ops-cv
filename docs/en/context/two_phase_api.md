# Two-phase API

Based on the single-operator API execution mode, calling an operator API typically follows a "two-phase" pattern, as shown below:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-phase interface aclxxXxxGetWorkspaceSize to calculate the amount of workspace memory required for this API call. After obtaining the required workspaceSize, allocate NPU memory based on workspaceSize, and then call the second-phase interface aclxxXxx to perform the computation.

Here, "aclxx" represents the operator interface prefix, such as aclnn, and "Xxx" represents the corresponding operator type, such as the Add operator.

> **NOTE**
>
>- The workspace refers to the temporary memory required by the API to complete computation on the AI processor, excluding input and output memory.
>- The second-phase interface aclxxXxx(...) cannot be called repeatedly. The following calling pattern will cause an exception:

  ```Cpp
  aclxxXxxGetWorkspaceSize(...)
  aclxxXxx(...)
  aclxxXxx(...)
  ```
