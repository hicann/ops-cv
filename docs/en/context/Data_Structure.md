# Data Structure

This chapter provides the basic data structures that depend on calling CANN operator APIs. **Developers do not need to pay attention to their internal implementation and can use them directly**.

Note that this basic data structure can be created through "Public Interface" in [Operator Acceleration Library](https://hiascend.com/document/redirect/CannCommunityOplist), such as aclCreateTensor.

- **aclTensor**

  A structure defined by the framework to manage and store tensor data (such as multi-dimensional data like vectors and matrices). This object can be created through the **aclCreateTensor** interface.

  ```cpp
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A structure defined by the framework to manage and store scalar data (that is, a single numerical value). This object can be created through the **aclCreateScalar** interface.

  ```cpp
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  An array structure defined by the framework to manage and store integer data. This object can be created through the **aclCreateIntArray** interface.

  ```cpp
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by the framework to manage and store float32 type data. This object can be created through the **aclCreateFloatArray** interface.

  ```cpp
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by the framework to manage and store Boolean type data. This object can be created through the **aclCreateBoolArray** interface.
    
  ```cpp
  typedef struct aclBoolArray aclBoolArray
  ```
    
- **aclTensorList**

  An array structure defined by the framework to manage and store multiple tensor data. This object can be created through the **aclCreateTensorList** interface.
    
  ```cpp
  typedef struct aclTensorList aclTensorList
  ```
    
- **aclScalarList**

  An array structure defined by the framework to manage and store scalar data. This object can be created through the **aclCreateScalarList** interface.

  ```cpp
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by the framework, which is a container used to execute operator calculations.

  Usually, when calling the first-stage interface aclxxXxxGetWorkspaceSize, the framework automatically creates an aclOpExecutor; after calling the second-stage interface aclxxXxx, the object is automatically released.

  ```cpp
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by the framework, used to manage and maintain the execution order of some asynchronous operations.
    
  ```cpp
  typedef void *aclrtStream
  ```
