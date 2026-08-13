# Conversion Relationship

When the **output aclTensor data type** of an API (such as aclnnAdd, aclnnMul, etc.) is inconsistent with the **calculation type deduced from the input data type**, the API internally converts the calculation result to the data type corresponding to the output type.

Data type conversion needs to satisfy the following rules. Conversions that do not satisfy the rules cannot be performed, and parameter validation will fail when calling the API.

- Floating-point types: ACL\_FLOAT16, ACL\_FLOAT, ACL\_DOUBLE, ACL\_BF16.
- Integer types: ACL\_INT8, ACL\_UINT8, ACL\_INT16, ACL\_UINT16, ACL\_INT32, ACL\_UINT32, ACL\_INT64, ACL\_UINT64.
- Complex types: ACL\_COMPLEX64, ACL\_COMPLEX128.
- Integer types can be converted between each other and also support conversion to floating-point and complex types.
- Floating-point types can be converted between each other and also support conversion to complex types.
- Complex types can be converted between each other.
- BOOL supports conversion to integer, floating-point, and complex types.

Except for the above scenarios, other conversion scenarios are not supported.
