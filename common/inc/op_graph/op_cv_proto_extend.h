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
 * \file op_cv_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_CV_PROTO_EXTEND_H_
#define OPS_OP_CV_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs Col2ImV2 for each batch entry. \n

* @par Inputs:
* @li x: The Col Tensor. 3-D, shape: `(n, c*kernel_h*kernel_w, ho*wo)`.
where ho/wo is do = (output_d + 2*padding_d - dilation_d*(kernel_d - 1) - 1)//stride_d + 1.
* @li output_size: The img shape Tensor. 1-D, shape:`(2)`, value: (output_h, output_w).
* @li kernel_shape: The kernel size Tensor. 1-D , value: `(kernel_h, kernel_w)`, the shape of kernel in convolution.  \n

* @par Outputs:
* y: The img Tensor. 4-D, shape: `(n, c, output_h, output_w)`. \n

* @par Attributes:

* @li dilation: ListInt, value: `(dilation_h, dilation_w)`, the dilation in convolution.
* @li padding: ListInt, value: `(padding_h, padding_w)`, the dilation in convolution.
* @li stride:  ListInt, value: `(stride_h, stride_w)`, the dilation in convolution.  \n

* @par Third-party framework compatibility
* Compatible with ONNX Col2Im operator.
*/
REG_OP(Col2ImV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(output_size, TensorType({DT_INT32, DT_INT32}))
    .INPUT(kernel_size, TensorType({DT_INT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(dilation, ListInt)
    .REQUIRED_ATTR(padding, ListInt)
    .REQUIRED_ATTR(stride, ListInt)
    .OP_END_FACTORY_REG(Col2ImV2)

/**
* @brief First calculate the minimum closure area of the two boxes, IoU,
* the proportion of the closed area that does not belong to the two boxes in the closure area,
* and finally subtract this proportion from IoU to get GIoU . \n

* @par Inputs:
* Two inputs, including:
* @li bboxes: Bounding boxes, a 2D Tensor of type float16 or float32 with
* shape (N, 4). "N" indicates the number of bounding boxes, and the value
* "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
* @li gtboxes: Ground-truth boxes, a 2D Tensor of type float16 or float32
* with shape (M, 4). "M" indicates the number of ground truth boxes, and
* the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h] . \n

* @par Attributes:
* @li trans: An optional bool, true for 'xywh', false for 'xyxy'.
* @li is_cross: An optional bool, control whether the output shape is [M, N] or [1, N]
* @li mode: Computation mode, a character string with the value range of [iou, iof] . \n

* @par Outputs:
* overlap: A 2D Tensor of type float16 or float32 with shape [M, N] or [1, N],
* specifying the IoU or IoF ratio . \n

* @attention Constraints:
* Only computation of float16 data is supported. To avoid overflow, the input
* length and width are scaled by 0.2 internally.
*/
REG_OP(GIoU)
    .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gtboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(overlap, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .ATTR(is_cross, Bool, true)
    .ATTR(mode, String, "iou")
    .OP_END_FACTORY_REG(GIoU)

/**
*@brief RotatedIou . \n

* @par Inputs:
*@li boxes : data of grad increment, a 3D Tensor of type float32 with
* shape (B, 5, N). "N" indicates the number of boxes, and the value
* "5" refers to [x1, y1, x2, y2, theta] or [x, y, w, h, theta].
* @li query_boxes: Bounding boxes, a 3D Tensor of type float32 with
* shape (B, 5, K). "K" indicates the number of boxes, and the value
* "5" refers to [x1, y1, x2, y2, theta] or [x, y, w, h, theta].

* @par Attributes:
*@li trans: An optional attr, true for 'xyxyt', false for 'xywht'.
* @li mode: An optional attr, a character string with the value range of ['iou', 'iof'],
* only support 'iou' now.
*@li is_cross: Cross calculation when it is True, and one-to-one calculation when it is False.
*@li v_threshold: An optional attr, provide condition relaxation for intersection calculation.
* @li e_threshold: An optional attr, provide condition relaxation for intersection calculation.

*@par Outputs:
* iou: A 3D Tensor of float32 with shape [B, N, K].

*@attention Constraints:
* In each batch, the invalid box cannot appear before the valid box.
*/
REG_OP(RotatedIou)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(query_boxes, TensorType({DT_FLOAT}))
    .OUTPUT(iou, TensorType({DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .ATTR(mode, String, "iou")
    .ATTR(is_cross, Bool, true)
    .ATTR(v_threshold, Float, 0)
    .ATTR(e_threshold, Float, 0)
    .OP_END_FACTORY_REG(RotatedIou)

/**
* @brief Resize the input tensor. \n
currently, only support resize image tensor using nearest neighbor and linear interpolation.

* @par Inputs:
* Input x must be a 4-D tensor. Inputs include: \n
* @li x: A Tensor. Must be one of the following types: uint8, int8, int16, \n
int32, int64, float16, float, double. 4-D with shape [batch, height, width, channels] \n
or shape [batch, channels, height, width].
* @li roi: A 1-D float Tensor. Only takes effect when attr coordinate_transformation_mode \n
is "tf_crop_and_resize". Must be one of the following types: float16, float, double.
* @li scales: A 1-D float Tensor, the scale array along each dimension, Only one of \n
'scales' and 'sizes' can be specified. Must be float type.
* @li sizes: A 1-D int64 Tensor, The size of the output tensor. Only one of \n
'scales' and 'sizes' can be specified.  If 'size' is specified, then set scales \n
to empty data (zero shape) in this operator's input list. Must be one of \n
the following types: int32, int64.

* @par Attributes:
* @li coordinate_transformation_mode: An optional String. how to transform \n
the coordinate in the resized tensor to the coordinate in the original tensor. \n
options: pytorch_half_pixel, align_corners, asymmetric, \n
tf_crop_and_resize.
* @li cubic_coeff_a: An optional Float. Defaults to -0.75, only used in cubic interpolation. \n
other optional: -0.5
* @li exclude_outside: An optional Int. Defaults to 0, If set to 1, the weight of sampling \n
locations outside the tensor will be set to 0 and the weight will be renormalized \n
so that their sum is 1.0.
* @li extrapolation_value: An optional Float. Defaults to 0.0f. When coordinate_transformation_mode \n
is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], \n
this value is used as the corresponding output value.
* @li mode: An optional String. Defaults to nearest. Three interpolation modes: nearest (default), \n
linear and cubic.
* @li nearest_mode: An optional String. Defaults to round_prefer_floor. Four modes: round_prefer_floor, \n
round_prefer_ceil, floor, ceil. Only used by nearest interpolation.

* @par Outputs:
* y: A Tensor. Has the same type as x.

* @attention Constraints: \n
* Input x must be a 4-D tensor.

* @par Third-party framework compatibility
* Compatible with tensorflow ResizeNearestNeighborV2 operator.
*/

REG_OP(Resize)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(roi, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sizes, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y,
            TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .OP_END_FACTORY_REG(Resize)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded
to be of length max_output_size. Defaults to false . \n

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size.
*@li valid_outputs: A 0-D integer tensor representing the number of valid
elements in selected_indices, with the valid elements appearing first . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionV4 operator.
*/

REG_OP(NonMaxSuppressionV4)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OUTPUT(valid_outputs, TensorType({DT_INT32}))
    .ATTR(pad_to_max_output_size, Bool, false)
    .OP_END_FACTORY_REG(NonMaxSuppressionV4)

/**
*@brief Normalizes data. It is called Region on YOLO v2 and Yolo on YOLO v3 . \n

*@par Inputs:
*x: An NCHW tensor of type float16 or float32. The data is with shape (N,
* boxes*(coords+obj+classes), H, W) . \n

*@par Attributes:
*@li boxes: An optional int32, specifying the number of anchor boxes. Defaults to "3".
*@li coords: An int32, specifying the number of parameters required for locating an object. Defaults to "4".
*@li classes: An int32, specifying the number of prediction classes. Defaults to "80".
*@li yolo_version: A string, specifying the YOLO version, either "V2" or "V3". Defaults to "V3"
*@li softmax: A bool, specifying whether to perform softmax. Defaults to "false".
*@li background: A bool. Defaults to "false".
*@li softmaxtree: A bool. Defaults to "false" . \n

*@par Outputs:
*@li coord_data: Specifies the coordinates of a detected box.
*@li obj_prob: Specifies the confidence.
*@li classes_prob: Specifies the prediction classes . \n
*/
#ifndef OPS_PROTO_DEF_YOLO
#define OPS_PROTO_DEF_YOLO
REG_OP(Yolo)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(coord_data, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(obj_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(classes_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(yolo_version, String, "V3")
    .ATTR(softmax, Bool, false)
    .ATTR(background, Bool, false)
    .ATTR(softmaxtree, Bool, false)
    .OP_END_FACTORY_REG(Yolo)
#endif

/**
*@brief Normalizes data. It is called Region on YOLO v2 and Yolo on YOLO v3 . \n

*@par Inputs:
*x: An NCHW tensor of type float16 or float32 . \n

*@par Attributes:
*@li boxes: An optional int32, specifying the number of anchor boxes. Defaults to "3".
*@li coords: An int32, specifying the number of parameters required for locating an object. Defaults to "4".
*@li classes: An int32, specifying the number of prediction classes. Defaults to "80".
*@li yolo_version: A string, specifying the YOLO version. Defaults to "V5"
*@li softmax: A bool, specifying whether to perform softmax. Defaults to "false".
*@li background: A bool. Defaults to "false".
*@li softmaxtree: A bool. Defaults to "false" . \n

*@par Outputs:
*@li coord_data: Specifies the coordinates of a detected box.
*@li obj_prob: Specifies the confidence.
*@li classes_prob: Specifies the prediction classes . \n
*/
#ifndef OPS_PROTO_DEF_YOLOPREDETECTION
#define OPS_PROTO_DEF_YOLOPREDETECTION
REG_OP(YoloPreDetection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(coord_data, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(obj_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(classes_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(yolo_version, String, "V5")
    .ATTR(softmax, Bool, false)
    .ATTR(background, Bool, false)
    .ATTR(softmaxtree, Bool, false)
    .OP_END_FACTORY_REG(YoloPreDetection)
#endif

/**
*@brief Performs YOLO V5 detection . \n

*@par Inputs:
*@li x: Dynamic inputs, the outputs of operator Yolo at the preceding layer.
*@li img_info: A float16 or float32, describing the image information . \n

*@par Attributes:
*@li biases: A required float. "biases = Number of Yolo operators at the preceding layer x 2 x boxes"
*@li boxes: A required int32, specifying the number of anchor boxes predicted for each Yolo layer.
*@li coords: Specifies the number of coordinate parameters. Must be 4.
*@li classes: A required int32, specifying the number of classes to be predicted.
*@li relative: An optional bool. Defaults to and must be "true".
*@li obj_threshold: A required float, specifying the confidence threshold for box filtering.
*@li post_nms_topn: An optional int32. This attribute is reserved.
*@li score_threshold: A required float, specifying the class score threshold for box filtering.
*@li iou_threshold: A required float, specifying the intersection-over-union (IOU) threshold for box filtering.\n
*@li pre_nms_topn: An optional int, specifying the number of boxes for non-maximum suppression (NMS). Defaults to "512".
*@li N: An optional int. Defaults to "10".
*@li resize_origin_img_to_net: An optional bool. Defaults to "false".
*@li out_box_dim: An optional int. Defaults to "3".
*@li alpha: An optional float. Defaults to "2.0".

*@par Outputs:
*@li box_out: A tensor of type float16 or float32, describing the information of each output box.
*@li box_out_num: A tensor of type int32, specifying the number of output boxes . \n
*/
#ifndef OPS_PROTO_DEF_YOLOV5DETECTIONOUTPUT
#define OPS_PROTO_DEF_YOLOV5DETECTIONOUTPUT
REG_OP(YoloV5DetectionOutput)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(biases, ListFloat)
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(relative, Bool, true)
    .ATTR(obj_threshold, Float, 0.5)
    .ATTR(post_nms_topn, Int, 512)
    .ATTR(score_threshold, Float, 0.5)
    .ATTR(iou_threshold, Float, 0.45f)
    .ATTR(pre_nms_topn, Int, 512)
    .ATTR(N, Int, 10)
    .ATTR(resize_origin_img_to_net, Bool, false)
    .ATTR(out_box_dim, Int, 3)
    .ATTR(alpha, Float, 2.0)
    .OUTPUT(box_out, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(box_out_num, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(YoloV5DetectionOutput)
#endif

/**
*@brief Performs YOLO V3 detection output V2 . \n

*@par Inputs:
*@li x: Dynamic inputs, the outputs of operator Yolo at the preceding layer . \n

*@par Attributes:
*@li biases: A float list. "biases = Number of Yolo operators at the preceding layer x 2 x boxes"
*@li boxes: A required int, specifying the number of anchor boxes predicted for each Yolo layer.
*@li coords: Specifies the number of coordinate parameters. Must be 4.
*@li classes: A required int, specifying the number of classes to be predicted.
*@li relative: An optional bool. Defaults to and must be "true".
*@li obj_threshold: A required float, specifying the confidence threshold for box filtering.
*@li post_nms_topn: An optional int. This attribute is reserved.
*@li score_threshold: A required float, specifying the class score threshold for box filtering.
*@li iou_threshold: A required float, specifying the intersection-over-union (IOU) threshold for box filtering.
*@li pre_nms_topn: An optional int, specifying the number of boxes for non-maximum suppression (NMS). Defaults to "512".
*@li N: An optional int. Defaults to "10".
*@li resize_origin_img_to_net: An optional bool. Defaults to "false".
*@li out_box_dim: An optional int. Defaults to "3".

*@par Outputs:
*@li box_out: A tensor of type float16 or float32, describing the information of each output box.
*@li box_out_num: A tensor of type int32, specifying the number of output boxes . \n
*/
#ifndef OPS_PROTO_DEF_YOLOV3DETECTIONOUTPUTV2
#define OPS_PROTO_DEF_YOLOV3DETECTIONOUTPUTV2
REG_OP(YoloV3DetectionOutputV2)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(biases, ListFloat)
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(relative, Bool, true)
    .ATTR(obj_threshold, Float, 0.5)
    .ATTR(post_nms_topn, Int, 512)
    .ATTR(score_threshold, Float, 0.5)
    .ATTR(iou_threshold, Float, 0.45f)
    .ATTR(pre_nms_topn, Int, 512)
    .ATTR(N, Int, 10)
    .ATTR(resize_origin_img_to_net, Bool, false)
    .ATTR(out_box_dim, Int, 3)
    .OUTPUT(box_out, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(box_out_num, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(YoloV3DetectionOutputV2)
#endif

/**
*@brief Generates bounding boxes based on "anchor_boxes" and "gt_bboxes" . \n

*@par Inputs:
*@li anchor_boxes: anchor sample boxes of origin image
*@li gt_bboxes: ground truth boxes
*@li stride: stride . \n

*@par Attributes:
*performance_mode: select performance mode, "high_precision" or "high_performance".

*@par Outputs:
*encoded_bboxes: Bboxes generated based on "anchor_boxes" and "gt_bboxes". Have the
* same format and type as "anchor_boxes".
*/
#ifndef OPS_PROTO_DEF_YOLOBOXESENCODE
#define OPS_PROTO_DEF_YOLOBOXESENCODE
REG_OP(YoloBoxesEncode)
    .INPUT(anchor_boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gt_bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(stride, TensorType({DT_INT32}))
    .ATTR(performance_mode, String, "high_precision")
    .OUTPUT(encoded_bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(YoloBoxesEncode)
#endif

/**
* @brief Generates bounding boxes based on "priors" and "bboxes".
* It is a customized yolox operator . \n

* @par Inputs:
* Two inputs, including:
* @li priors: prior sample boxes of origin image
* A 2D Tensor of type float32 or float16 with shape (N, 4).
* @li bboxes_input: bboxes predicted by the model. A 2D Tensor of type float32 or float16 with shape (B, N, 4) . \n

* @par Outputs:
* bboxes_output: Bboxes generated based on "priors" and "bboxes_input". Have the same format
* and type as "bboxes_input".
*/
#ifndef OPS_PROTO_DEF_YOLOXBOUNDINGBOXDECODE
#define OPS_PROTO_DEF_YOLOXBOUNDINGBOXDECODE
REG_OP(YoloxBoundingBoxDecode)
    .INPUT(priors, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(decoded_bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(YoloxBoundingBoxDecode)
#endif

/**
*@brief Converts ARGB channel . \n

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16.

*@par Outputs:
*y: A Tensor. Must be one of the following types: int16. \n
*/
#ifndef OPS_PROTO_DEF_TRANSARGB
#define OPS_PROTO_DEF_TRANSARGB
REG_OP(TransArgb)
    .INPUT(x, "T1")
    .OUTPUT(y, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_INT16}))
    .OP_END_FACTORY_REG(TransArgb)
#endif
} // namespace ge

#endif