# Data Structure

This chapter provides the basic data structures required for calling CANN operator APIs. **Developers do not need to focus on their internal implementation and can use them directly.**

Note: These basic data structures can be created through the "public interfaces" in [*Operator Acceleration Library*](https://gitcode.com/cann/opbase/blob/master/docs/en/api/nnopbase/aclnn/00_aclnn_api_list.md), such as aclCreateTensor.

- **aclTensor**

  A structure defined by the framework for managing and storing tensor data (such as vectors, matrices, and other multi-dimensional data). This object can be created through the **aclCreateTensor** interface.

  ```cpp
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A framework-defined structure for managing and storing scalar data (i.e., a single numeric value). The object can be created through the **aclCreateScalar** interface.

  ```cpp
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  A framework-defined array structure for managing and storing integer data. The object can be created through the **aclCreateIntArray** interface.

  ```cpp
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by the framework for managing and storing float32 data. This object can be created through the **aclCreateFloatArray** interface.

  ```cpp
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by the framework for managing and storing Boolean data. The object can be created through the **aclCreateBoolArray** interface.

  ```cpp
  typedef struct aclBoolArray aclBoolArray
  ```

- **aclTensorList**

  An array structure defined by the framework for managing and storing multiple tensor data. This object can be created through the **aclCreateTensorList** interface.

  ```cpp
  typedef struct aclTensorList aclTensorList
  ```

- **aclScalarList**

  An array structure defined by the framework for managing and storing scalar data. This object can be created through the **aclCreateScalarList** interface.

  ```cpp
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by the framework, serving as a container for executing operator computations.

  Typically, when calling the first-phase operator interface aclxxXxxGetWorkspaceSize, the framework automatically creates an aclOpExecutor; after calling the second-phase interface aclxxXxx, the object is automatically released.

  ```cpp
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by the framework, used to manage and maintain the execution order of some asynchronous operations.

  ```cpp
  typedef void *aclrtStream
  ```
