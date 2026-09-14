#ifndef TRI_ATT_FA_OFFICIAL_PROBE_TILING_H
#define TRI_ATT_FA_OFFICIAL_PROBE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TriAttFaOfficialProbeTilingData)
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, pseStride);
TILING_DATA_FIELD_DEF(uint32_t, bias1Stride);
TILING_DATA_FIELD_DEF(uint32_t, bias1Rows);
TILING_DATA_FIELD_DEF(uint32_t, bias2BatchShared);
TILING_DATA_FIELD_DEF(uint32_t, bias2GroupsPerBatch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TriAttFaOfficialProbe, TriAttFaOfficialProbeTilingData)
}  // namespace optiling

#endif
