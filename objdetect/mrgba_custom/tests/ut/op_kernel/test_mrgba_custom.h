/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _FAST_OP_TEST_MRGBA_CUSTOM_TILING_H_
#define _FAST_OP_TEST_MRGBA_CUSTOM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct MrgbaCustomTilingData {
  uint32_t alphaLen = 0;
};

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                \
  MrgbaCustomTilingData tilingData;                                                     \
  INIT_TILING_DATA(MrgbaCustomTilingData, tilingDataPointer, tilingPointer);            \
  (tilingData).alphaLen = tilingDataPointer->alphaLen;
#endif // _FAST_OP_TEST_MRGBA_CUSTOM_TILING_H_
