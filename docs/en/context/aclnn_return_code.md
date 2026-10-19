# aclnn Return Codes

When calling the aclnn API, common API return codes are shown in [Table 1](#table1).
For abnormal status code values, obtain the exception information through the aclGetRecentErrMsg API ([Runtime API](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi)). You can troubleshoot the problem based on the error message or contact technical support.

**Table 1**  Return Codes <a id="table1"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | Success. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter validation error; an illegal nullptr exists in the parameters. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter validation error, for example the two input data types do not satisfy the input type deduction relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | The API internally calls an NPU runtime interface abnormally. |
| ACLNN_ERR_INNER_XXX | 561xxx | An exception occurs inside the API. |

More details about the ACLNN_ERR_INNER_XXX status codes are shown in [Table 2](#table2).

**Table 2**  Abnormal Status Codes <a id="table2"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER |   561000     |    Internal exception: An exception occurs in the API.     |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR     |  561001      |  Internal exception: An error occurs during output shape inference inside the API.       |
| ACLNN_ERR_INNER_TILING_ERROR     |   561002     |   Internal exception: An exception occurs when the API performs NPU kernel tiling internally.     |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR   |  561003  |  Internal exception: An exception occurs when the API searches for the NPU kernel internally (possibly because the operator binary package is not installed).    |
| ACLNN_ERR_INNER_CREATE_EXECUTOR     |    561101    |   Internal exception: The API fails to create aclOpExecutor internally (possibly because of an OS exception).      |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR     |  561102      |   Internal exception: The API does not call uniqueExecutor ReleaseTo internally.      |
| ACLNN_ERR_INNER_NULLPTR            |   561103     |    Internal exception: An exception occurs inside the aclnn API; a nullptr exception occurs.     |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE     |   561104     |   Internal exception: An exception occurs inside the aclnn API; the number of operator attributes is abnormal.      |
| ACLNN_ERR_INNER_KEY_CONFILICT (Deprecated)     |   561105     |    **Deprecated. Use the latest ACLNN_ERR_INNER_KEY_CONFLICT.**     |
| ACLNN_ERR_INNER_KEY_CONFLICT |   561105     |   Internal exception: An exception occurs inside the aclnn API; the hash key matched by the operator's kernel conflicts. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE     |   561106     |   Internal exception: An exception occurs inside the aclnn API; the operator's implementation mode parameter is incorrect.    |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND     |  561107     |  Internal exception: An exception occurs inside the aclnn API; the environment variable ASCEND_OPP_PATH that needs to be configured is not detected.       |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED     |  561108      |   Internal exception: An exception occurs inside the aclnn API; loading the operator info JSON file from the operator kernel library fails.      |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND     |   561109      |   Internal exception: An exception occurs inside the aclnn API; loading a field of the operator info JSON file from the operator kernel library fails.      |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID     |  561110     |     Internal exception: An exception occurs inside the aclnn API; the format of the operator info JSON file in the operator kernel library is set to an illegal value.    |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID     |    561111      |     Internal exception: An exception occurs inside the aclnn API; the dtype of the operator info JSON file in the operator kernel library is set to an illegal value.    |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND     |   561112    |    Internal exception: An exception occurs inside the aclnn API; the operator's binary kernel library is not loaded.     |
| ACLNN_ERR_INNER_OP_FILE_INVALID     |  561113     |   Internal exception: An exception occurs inside the aclnn API; an exception occurs when loading operator JSON file fields.   |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND     |  561114      |  Internal exception: An exception occurs inside the aclnn API; the number of operator attributes is inconsistent with the operator info JSON, exceeding the number of attrs specified in the JSON.       |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH     |   561115      |   Internal exception: An exception occurs inside the aclnn API; the number of operator attributes is inconsistent with the operator info JSON, fewer than the number of attrs specified in the JSON.      |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE     |  561116     |   Internal exception: An exception occurs inside the aclnn API; the number of operator inputs exceeds the limit of 32.      |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL     |  561117     |  Internal exception: An exception occurs inside the aclnn API; the operator info JSON file has missing descriptions.       |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID     |  561118     |    Internal exception: An exception occurs inside the aclnn API; an exception occurs when parsing the workspace information in the static binary JSON file.     |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID     |  561119      |    Internal exception: An exception occurs inside the aclnn API; an exception occurs when parsing the core count usage information in the static binary JSON file.  |
