/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stub_ops.h
 * \brief
 */

#ifndef CV_COMMON_STUB_OPS_H
#define CV_COMMON_STUB_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Input data for other operators. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the Caffe operator Data.
*/
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)

/**
*@brief Cast a tensor from src data type to dst data type.

*@par Inputs:
*One input:
* x:An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32,
uint32, uint8, bfloat16, uint1, int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8,
quint8, qint16, quint16, qint32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

*@par Attributes:
*dst_type: A required attribute of type int32, specifying the dst data type.

*@par Outputs:
*y:An ND Tensor with same shape as x, and data type is specified by dst_type.

*@attention Constraints:
* @li In the scenario where the data type is converted from float16 to int16: \n
*     If the input data contains inf, inf is converted into the maximum value of int16. \n
*     If the input data contains -inf, -inf is converted into the minimum value of int16. \n
* @li In the scenarios where the data type is converted from INT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and
from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648,
2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648,
2147483647).
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL,          DT_FLOAT16,     DT_FLOAT,      DT_INT8,      DT_INT32,    DT_UINT32,
                          DT_UINT8,         DT_INT64,       DT_UINT64,     DT_INT16,     DT_UINT16,   DT_DOUBLE,
                          DT_COMPLEX64,     DT_COMPLEX128,  DT_QINT8,      DT_QUINT8,    DT_QINT16,   DT_QUINT16,
                          DT_QINT32,        DT_BF16,        DT_UINT1,      DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL,        DT_FLOAT16,    DT_FLOAT,     DT_INT8,     DT_INT32,       DT_UINT32,
                           DT_UINT8,       DT_INT64,      DT_UINT64,    DT_INT16,    DT_UINT16,      DT_DOUBLE,
                           DT_COMPLEX64,   DT_COMPLEX128, DT_QINT8,     DT_QUINT8,   DT_QINT16,      DT_QUINT16,
                           DT_QINT32,      DT_BF16,       DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
Operator Const has the same definition as operator Constant. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: A constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Const).OUTPUT(y, TensorType::ALL()).ATTR(value, Tensor, Tensor()).OP_END_FACTORY_REG(Const)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li concat_dim: Must be one of the IndexNumberType: int32, int64.
* Specifies the dimension along which to concatenate .
* @li x: Dynamic input.A ND Tensor.
* Must be one of the BasicType:
  complex128, complex64, double, float32, float16, int16, int32, int64, int8,
  qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
  bfloat16, complex32, bool. \n


* @par Attributes:
* N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x" .
  Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat. \n
*/
REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data.
\n

*@par Inputs:
*x: Original tensor. All data types are supported. \n

*@par Attributes:
*axes: List of ints indicating the dimensions to be inserted. Defaults to []. \n

*@par Outputs:
*y: Reshape tensor with same data as input. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.
*/

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)

/**
* @brief Permutes the dimensions according to perm.
     The returned tensor's dimension i will correspond to the input dimension perm[i].

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bool, hifloat8, float8_e5m2,
* float8_e4m3fn, and the maximum dimension should not exceed 8 dimensions,
* and the shape should be consistent with output.
* @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x", the value
* should be within the range of [0, number of dimensions for self -1].

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                          DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                          DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                           DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                           DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                           DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

/**
* @brief Returns x1 * x2 element-wise.
* y = x1 * x2. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* @li x2: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* The shape of x1 and x2 must meet the requirements of the broadcast relationship.

* @par Outputs:
* y: A ND tensor. Must be one of the following types: bool, float16, float32, float64, bfloat16,
* uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.

* @attention Constraints:
* "x1" and "x2" have incompatible shapes or types.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16,
                              DT_INT32, DT_INT64, DT_BF16, DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16,
                              DT_INT32, DT_INT64, DT_BF16, DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Mul)

/**
*@brief Return a tensor with the same shape and contents as input. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Outputs:
*y: A tensor with the same shape、data type and contents as input. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Identity.
*/
REG_OP(Identity).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).OP_END_FACTORY_REG(Identity)

/**
*@brief Returns the shape of a tensor. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor. The shape of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Shape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Creates a tensor with the given "shape" and "dtype". \n

*@par Inputs:
*shape: The shape of the output tensor. \n

*@par Attributes:
*@li dtype: Optional. The data type of the output tensor. Defaults to "int32".
*@li init: An optional bool. If true, initializes the returned tensor with the default value of "dtype". Defaults to
"false". \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Empty.
*/
REG_OP(Empty)
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE, DT_BF16, DT_STRING, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(dtype, Int, DT_INT32)
    .ATTR(init, Bool, false)
    .OP_END_FACTORY_REG(Empty)
} // namespace ge

#endif // CV_COMMON_STUB_OPS_H
