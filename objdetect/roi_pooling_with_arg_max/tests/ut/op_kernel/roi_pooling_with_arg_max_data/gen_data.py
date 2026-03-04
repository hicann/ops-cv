#!/usr/bin/python3
import sys
import os
import numpy as np
import math

input0_x = np.random.randn(2, 16, 25, 42).astype(np.float32)
input0_roi = np.array([[0, 0, 0, 10, 10], [1, 10, 10, 20, 20]], dtype=np.float32)

case0_params = {
    "input_x": input0_x,
    "input_roi": input0_roi,
    "pooled_height": 3,
    "pooled_width": 3,
    "spatial_scale_h": 1.0,
    "spatial_scale_w": 1.0,
}


def roi_pooling_with_arg_max_golden(feature_map, rois, pooled_height, pooled_width, spatial_scale_h, spatial_scale_w):
    # 获取输入参数
    batch_size, channels, height, width = feature_map.shape
    num_rois = rois.shape[0]

    # 初始化输出和argmax数组
    output = np.zeros((num_rois, channels, pooled_height, pooled_width), dtype=feature_map.dtype)
    argmax = -np.ones((num_rois, channels, pooled_height, pooled_width), dtype=np.int32)

    for n in range(num_rois):
        roi = rois[n]
        roi_batch_ind = int(roi[0])
        x1, y1, x2, y2 = roi[1], roi[2], roi[3], roi[4]

        # 转换到特征图坐标
        roi_x1 = x1 * spatial_scale_w
        roi_y1 = y1 * spatial_scale_h
        roi_x2 = (x2 + 1) * spatial_scale_w  # CUDA中的+1处理
        roi_y2 = (y2 + 1) * spatial_scale_h

        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        # 处理无效ROI区域
        if roi_w <= 0 or roi_h <= 0:
            output[n] = 0
            argmax[n] = -1
            continue

        bin_size_w = roi_w / pooled_width
        bin_size_h = roi_h / pooled_height

        for c in range(channels):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    # 计算当前池化单元对应的区域
                    bin_x1 = pw * bin_size_w + roi_x1
                    bin_x1 = int(math.floor(bin_x1))
                    bin_x2 = (pw + 1) * bin_size_w + roi_x1
                    bin_x2 = int(math.ceil(bin_x2))

                    bin_y1 = ph * bin_size_h + roi_y1
                    bin_y1 = int(math.floor(bin_y1))
                    bin_y2 = (ph + 1) * bin_size_h + roi_y1
                    bin_y2 = int(math.ceil(bin_y2))

                    # 裁剪到有效范围
                    bin_x1 = min(max(bin_x1, 0), width)
                    bin_x2 = min(max(bin_x2, 0), width)
                    bin_y1 = min(max(bin_y1, 0), height)
                    bin_y2 = min(max(bin_y2, 0), height)

                    # 检查是否为空区域
                    if (bin_x2 <= bin_x1) or (bin_y2 <= bin_y1):
                        output[n, c, ph, pw] = 0
                        argmax[n, c, ph, pw] = -1
                        continue
                    # 提取特征区域
                    input_region = feature_map[roi_batch_ind, c, bin_y1:bin_y2, bin_x1:bin_x2]
                    if input_region.size == 0:
                        output_val = 0
                        argmax_val = -1
                    else:
                        output_val = input_region.max()
                        # 找到所有最大值的位置并取第一个
                        max_positions = np.argwhere(input_region == output_val)
                        if max_positions.size == 0:
                            output_val = 0
                            argmax_val = -1
                        else:
                            # 获取第一个最大值的位置
                            h_rel, w_rel = max_positions[0]
                            h_abs = bin_y1 + h_rel
                            w_abs = bin_x1 + w_rel
                            argmax_val = h_abs * width + w_abs
                    output[n, c, ph, pw] = output_val
                    argmax[n, c, ph, pw] = argmax_val

    return output, argmax


def gen_golden_data(case_name):
    if case_name == "test_simt_float32_case0":
        case_params = case0_params
    else:
        print(f"Unknown case: {case_name}")
        return
    
    x = case_params["input_x"]
    rois = case_params["input_roi"]
    pooled_height = case_params["pooled_height"]
    pooled_width = case_params["pooled_width"]
    spatial_scale_h = case_params["spatial_scale_h"]
    spatial_scale_w = case_params["spatial_scale_w"]
    
    x.tofile("./x.bin")
    rois.tofile("./rois.bin")
    res, res_idx = roi_pooling_with_arg_max_golden(
        x, rois, pooled_height, pooled_width, spatial_scale_h, spatial_scale_w)
    res.tofile("./y_golden.bin")
    res_idx.tofile("./argmax_golden.bin")
    print(f"Generated golden data for {case_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_data.py <case_name>")
        sys.exit(1)
    gen_golden_data(sys.argv[1])
