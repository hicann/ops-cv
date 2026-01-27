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
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Cast a tensor from src data type to dst data type.

*@par Inputs:
*One input:
* x:An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8, bfloat16, uint1,
   int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32,
   hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

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
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483647).
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1,
                          DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,
                           DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
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
REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

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
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

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
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)
}  // namespace ge

#endif  // CV_COMMON_STUB_OPS_H
