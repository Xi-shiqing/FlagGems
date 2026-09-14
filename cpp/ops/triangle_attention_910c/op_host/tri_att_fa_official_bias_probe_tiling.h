#ifndef TRI_ATT_FA_OFFICIAL_BIAS_PROBE_TILING_H
#define TRI_ATT_FA_OFFICIAL_BIAS_PROBE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/matmul/matmul_tilingdata.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TriAttFaOfficialBiasProbeTilingData)
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
TILING_DATA_FIELD_DEF(uint32_t, qHeadStride);
TILING_DATA_FIELD_DEF(uint32_t, kvHeadStride);
TILING_DATA_FIELD_DEF(uint32_t, qTileSize);
TILING_DATA_FIELD_DEF(uint32_t, kvStackSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1FullFull);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1FullTail);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TailFull);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TailTail);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2FullFull);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2FullTail);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TailFull);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TailTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TriAttFaOfficialBiasProbe, TriAttFaOfficialBiasProbeTilingData)
}  // namespace optiling

#endif
