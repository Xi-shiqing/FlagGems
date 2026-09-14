#ifndef TRI_ATT_QK_BIAS_FUSED_TILING_H
#define TRI_ATT_QK_BIAS_FUSED_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TriAttQkBiasFusedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, blockM);
TILING_DATA_FIELD_DEF(uint32_t, blockN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TriAttQkBiasFused, TriAttQkBiasFusedTilingData)
}  // namespace optiling

#endif  // TRI_ATT_QK_BIAS_FUSED_TILING_H
