#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {"g_io_u_grad": "g_io_u_grad_golden"},
}


def _f32(x):
    return np.float32(x)


def _tbe_g_io_u_grad_element(dy_val, b_x1, b_y1, b_x2, b_y2, g_x1, g_y1, g_x2, g_y2):
    """
    Manual GIoU gradient matching TBE common_iou_grad.py + giou_grad.py exactly.
    All intermediate ops use float32 and follow TBE's computation order.
    """
    eps = _f32(1e-9)
    dy_val = _f32(dy_val)
    b_x1, b_y1, b_x2, b_y2 = _f32(b_x1), _f32(b_y1), _f32(b_x2), _f32(b_y2)
    g_x1, g_y1, g_x2, g_y2 = _f32(g_x1), _f32(g_y1), _f32(g_x2), _f32(g_y2)

    # TBE move_in: compute w/h then clamp + eps
    b_w_raw = _f32(float(b_x2) - float(b_x1))
    b_h_raw = _f32(float(b_y2) - float(b_y1))
    g_w_raw = _f32(float(g_x2) - float(g_x1))
    g_h_raw = _f32(float(g_y2) - float(g_y1))

    b_w = _f32(max(float(b_w_raw), 0.0) + float(eps))
    b_h = _f32(max(float(b_h_raw), 0.0) + float(eps))
    g_w = _f32(max(float(g_w_raw), 0.0) + float(eps))
    g_h = _f32(max(float(g_h_raw), 0.0) + float(eps))

    # TBE update_forward_part: inter
    xlen = _f32(min(float(b_x2), float(g_x2)) - max(float(b_x1), float(g_x1)))
    ylen = _f32(min(float(b_y2), float(g_y2)) - max(float(b_y1), float(g_y1)))
    inter_x = _f32(max(float(xlen), 0.0))
    inter_y = _f32(max(float(ylen), 0.0))
    inter = _f32(float(inter_x) * float(inter_y))

    # TBE update_forward_part: union (NO eps)
    b_area = _f32(float(b_w) * float(b_h))
    g_area = _f32(float(g_w) * float(g_h))
    uni = _f32(float(b_area) + float(g_area))
    uni = _f32(float(uni) - float(inter))

    # TBE update_forward_part: cw/ch/enclose
    raw_cw = _f32(max(float(b_x2), float(g_x2)) - min(float(b_x1), float(g_x1)))
    raw_ch = _f32(max(float(b_y2), float(g_y2)) - min(float(b_y1), float(g_y1)))
    cw = _f32(max(float(raw_cw), 0.0) + float(eps))
    ch = _f32(max(float(raw_ch), 0.0) + float(eps))
    enclose = _f32(float(cw) * float(ch))
    enclose = _f32(float(enclose) + float(eps))

    # TBE GIoUGrad.update_backward_part (giou_grad.py lines 82-96):
    # dunion = dy/enclose - dy * (inter/union) / union
    tmp_a = _f32(float(dy_val) / float(enclose))
    tmp_b = _f32(float(inter) / float(uni))
    tmp_c = _f32(float(tmp_b) / float(uni))
    tmp_d = _f32(float(dy_val) * float(tmp_c))
    dunion = _f32(float(tmp_a) - float(tmp_d))

    # dinter = dy/union - dunion
    dinter = _f32(float(dy_val) / float(uni))
    dinter = _f32(float(dinter) - float(dunion))

    # denclose = -(union / enclose^2) * dy
    tmp_a2 = _f32(float(uni) / float(enclose))
    tmp_b2 = _f32(float(tmp_a2) / float(enclose))
    tmp_c2 = _f32(float(dy_val) * float(tmp_b2))
    denclose = _f32(-float(tmp_c2))

    # TBE inter_part: dxlen = dinter * inter_y, dylen = dinter * inter_x
    dxlen = _f32(float(dinter) * float(inter_y))
    dylen = _f32(float(dinter) * float(inter_x))

    # Initialize dx outputs to 0
    db_x1 = _f32(0.0)
    db_y1 = _f32(0.0)
    db_x2 = _f32(0.0)
    db_y2 = _f32(0.0)
    dg_x1 = _f32(0.0)
    dg_y1 = _f32(0.0)
    dg_x2 = _f32(0.0)
    dg_y2 = _f32(0.0)

    # TBE inter_part: x2 direction (vec_cmpv_lt + vec_cmpv_ge gate)
    if float(xlen) >= 0.0:
        if float(b_x2) < float(g_x2):
            db_x2 = _f32(float(db_x2) + float(dxlen))
        else:
            dg_x2 = _f32(float(dg_x2) + float(dxlen))

    # TBE inter_part: negate dxlen for x1 direction
    neg_dxlen = _f32(-float(dxlen))
    if float(xlen) >= 0.0:
        if float(b_x1) > float(g_x1):
            db_x1 = _f32(float(db_x1) + float(neg_dxlen))
        else:
            dg_x1 = _f32(float(dg_x1) + float(neg_dxlen))

    # TBE inter_part: y2 direction
    if float(ylen) >= 0.0:
        if float(b_y2) < float(g_y2):
            db_y2 = _f32(float(db_y2) + float(dylen))
        else:
            dg_y2 = _f32(float(dg_y2) + float(dylen))

    # TBE inter_part: negate dylen for y1 direction
    neg_dylen = _f32(-float(dylen))
    if float(ylen) >= 0.0:
        if float(b_y1) > float(g_y1):
            db_y1 = _f32(float(db_y1) + float(neg_dylen))
        else:
            dg_y1 = _f32(float(dg_y1) + float(neg_dylen))

    # TBE union_part: always distribute (no gating)
    # tmp_a = b_w * dunion (for dy direction)
    # tmp_b = b_h * dunion (for dx direction)
    bw_dunion = _f32(float(b_w) * float(dunion))
    bh_dunion = _f32(float(b_h) * float(dunion))
    db_x2 = _f32(float(db_x2) + float(bh_dunion))
    db_x1 = _f32(float(db_x1) - float(bh_dunion))
    db_y2 = _f32(float(db_y2) + float(bw_dunion))
    db_y1 = _f32(float(db_y1) - float(bw_dunion))

    gw_dunion = _f32(float(g_w) * float(dunion))
    gh_dunion = _f32(float(g_h) * float(dunion))
    dg_x2 = _f32(float(dg_x2) + float(gh_dunion))
    dg_x1 = _f32(float(dg_x1) - float(gh_dunion))
    dg_y2 = _f32(float(dg_y2) + float(gw_dunion))
    dg_y1 = _f32(float(dg_y1) - float(gw_dunion))

    # TBE GIoUGrad.enclose_part:
    dxlen_enc = _f32(float(denclose) * float(ch))
    dylen_enc = _f32(float(denclose) * float(cw))

    # max(b_x2, g_x2): vec_cmpv_gt(b_x2, g_x2)
    if float(b_x2) > float(g_x2):
        db_x2 = _f32(float(db_x2) + float(dxlen_enc))
    else:
        dg_x2 = _f32(float(dg_x2) + float(dxlen_enc))

    if float(b_y2) > float(g_y2):
        db_y2 = _f32(float(db_y2) + float(dylen_enc))
    else:
        dg_y2 = _f32(float(dg_y2) + float(dylen_enc))

    # min(b_x1, g_x1): negate then vec_cmpv_lt(b_x1, g_x1)
    neg_dxlen_enc = _f32(-float(dxlen_enc))
    neg_dylen_enc = _f32(-float(dylen_enc))

    if float(b_x1) < float(g_x1):
        db_x1 = _f32(float(db_x1) + float(neg_dxlen_enc))
    else:
        dg_x1 = _f32(float(dg_x1) + float(neg_dxlen_enc))

    if float(b_y1) < float(g_y1):
        db_y1 = _f32(float(db_y1) + float(neg_dylen_enc))
    else:
        dg_y1 = _f32(float(dg_y1) + float(neg_dylen_enc))

    return (float(db_x1), float(db_y1), float(db_x2), float(db_y2),
            float(dg_x1), float(dg_y1), float(dg_x2), float(dg_y2))


def g_io_u_grad_golden(dy, bboxes, gtboxes, *, trans=False, is_cross=True, mode="iou", **kwargs):
    if mode != "iou":
        raise ValueError(f"Unsupported mode='{mode}'. Only mode='iou' is supported.")

    eps = np.float32(1e-9)
    half = np.float32(0.5)
    orig_dtype = bboxes.dtype

    dy_f32 = dy.astype(np.float32) if dy.dtype.name in ('bfloat16', 'float16') else dy.copy()
    b_f32 = bboxes.astype(np.float32) if bboxes.dtype.name in ('bfloat16', 'float16') else bboxes.copy()
    g_f32 = gtboxes.astype(np.float32) if gtboxes.dtype.name in ('bfloat16', 'float16') else gtboxes.copy()

    n = dy_f32.shape[0]

    # TBE trans_true: xywh -> xyxy using RAW w (before clamp)
    if trans:
        raw_bw_half = b_f32[2] * half
        raw_bh_half = b_f32[3] * half
        b_xyxy = np.stack([
            b_f32[0] - raw_bw_half,
            b_f32[1] - raw_bh_half,
            b_f32[0] + raw_bw_half,
            b_f32[1] + raw_bh_half
        ], axis=0).astype(np.float32)
        raw_gw_half = g_f32[2] * half
        raw_gh_half = g_f32[3] * half
        g_xyxy = np.stack([
            g_f32[0] - raw_gw_half,
            g_f32[1] - raw_gh_half,
            g_f32[0] + raw_gw_half,
            g_f32[1] + raw_gh_half
        ], axis=0).astype(np.float32)
    else:
        b_xyxy = b_f32
        g_xyxy = g_f32

    db_xyxy = np.zeros((4, n), dtype=np.float32)
    dg_xyxy = np.zeros((4, n), dtype=np.float32)

    for i in range(n):
        grads = _tbe_g_io_u_grad_element(
            dy_f32[i],
            b_xyxy[0, i], b_xyxy[1, i], b_xyxy[2, i], b_xyxy[3, i],
            g_xyxy[0, i], g_xyxy[1, i], g_xyxy[2, i], g_xyxy[3, i]
        )
        db_xyxy[0, i] = grads[0]
        db_xyxy[1, i] = grads[1]
        db_xyxy[2, i] = grads[2]
        db_xyxy[3, i] = grads[3]
        dg_xyxy[0, i] = grads[4]
        dg_xyxy[1, i] = grads[5]
        dg_xyxy[2, i] = grads[6]
        dg_xyxy[3, i] = grads[7]

    if trans:
        dbboxes = np.stack([
            db_xyxy[0] + db_xyxy[2],
            db_xyxy[1] + db_xyxy[3],
            (db_xyxy[2] - db_xyxy[0]) * half,
            (db_xyxy[3] - db_xyxy[1]) * half
        ], axis=0).astype(np.float32)
        dgtboxes = np.stack([
            dg_xyxy[0] + dg_xyxy[2],
            dg_xyxy[1] + dg_xyxy[3],
            (dg_xyxy[2] - dg_xyxy[0]) * half,
            (dg_xyxy[3] - dg_xyxy[1]) * half
        ], axis=0).astype(np.float32)
    else:
        dbboxes = db_xyxy
        dgtboxes = dg_xyxy

    if orig_dtype.name in ('bfloat16', 'float16'):
        dbboxes = dbboxes.astype(np.float32)
        dgtboxes = dgtboxes.astype(np.float32)
        if orig_dtype.name == 'float16':
            dbboxes = dbboxes.astype(np.float16)
            dgtboxes = dgtboxes.astype(np.float16)
    else:
        dbboxes = dbboxes.astype(orig_dtype)
        dgtboxes = dgtboxes.astype(orig_dtype)

    return dbboxes, dgtboxes
