# Two-stage Interface

When calling operator APIs based on the single operator API execution mode, it is usually divided into "two stages", with the style as follows:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-stage interface aclxxXxxGetWorkspaceSize to calculate how much workspace memory is needed during this API call process. After obtaining the required workspaceSize, apply for NPU memory according to workspaceSize, and then call the second-stage interface aclxxXxx to execute the calculation.

Where "aclxx" represents the operator interface prefix, such as aclnn; and "Xxx" represents the corresponding operator type, such as the Add operator.

> Note:
>
>- workspace refers to the temporary memory required by the API to complete calculations on the AI processor in addition to input/output.
>- The second-stage interface aclxxXxx(...) cannot be called repeatedly. The following calling method will cause an exception:
>
>   ```Cpp     
>   aclxxXxxGetWorkspaceSize(...)  
>   aclxxXxx(...)   
>   aclxxXxx(...)
>   ```
