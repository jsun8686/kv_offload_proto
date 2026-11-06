/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * PoC adaptation: replace <securec.h> with a minimal shim so the header compiles
 * outside the full CANN securec environment. Only memset_s is used by the copied
 * headers below. Original source: hicann/hcomm include/hcomm_res_defs.h.
 */
#ifndef POC_SECUREC_SHIM_H
#define POC_SECUREC_SHIM_H

#include <cstring>
#include <cstddef>

#ifndef EOK
#define EOK 0
#endif

// securec.h 原生提供 errno_t（Linux 下为 int）；hcomm_res_defs.h 的 ThreadConfigInit 使用
#ifndef POC_HAVE_ERRNO_T
#define POC_HAVE_ERRNO_T
typedef int errno_t;
#endif

// PoC-only shim：hcomm_res_defs.h 的 InlineInit 函数使用。非安全审查产物，勿用于生产。
inline int poc_memset_s(void *dest, size_t destMax, int value, size_t count)
{
    if (dest == nullptr || count > destMax) {
        return -1;
    }
    (void)memset(dest, value, count);
    return EOK;
}

#define memset_s poc_memset_s

#endif  // POC_SECUREC_SHIM_H
