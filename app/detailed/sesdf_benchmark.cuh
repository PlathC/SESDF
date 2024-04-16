#ifndef SESDF_BENCHMARK_CUH
#define SESDF_BENCHMARK_CUH

#include <bcs/core/type.hpp>
#include <bcs/cuda/grid.cuh>
#include <bcs/cuda/memory.cuh>
#include <bcs/sesdf/data.cuh>

#include "timer.cuh"
#include "grid_benchmark.cuh"

namespace bcs::sesdf
{
    template<uint32_t MaxNeighborPerAtom>
    std::pair<ResultBuffer, DeviceBuffer> findCirclesBenchmark( AccelerationGridBenchmark & grid,
                                                                bcs::sesdf::SesdfContext &  sesContext,
                                                                uint2 *                     convexPatches,
                                                                bool                        graphics,
                                                                float &                     computeTime );

    template<uint32_t MaxNeighborPerAtom>
    std::pair<ResultBuffer, DeviceBuffer> findIntersectionsBenchmark( SesdfContext & sesContext,
                                                                      bool           graphics,
                                                                      float &        computeTime );

    template<uint8_t MaxIntersectionsPerCircle>
    ResultBuffer buildConicPatchesBenchmark( SesdfContext &       sesContext,
                                             const float4 * const intersectionsPositions,
                                             const int4 * const   intersectionAtomIds,
                                             bool                 graphics,
                                             float &              computeTime );

    void handleIntersectionSingularitiesBenchmark( AccelerationGridBenchmark & grid,
                                                   const SesdfContext &        sesContext,
                                                   const float4 * const        intersectionsPositions,
                                                   int4 *                      intersectionAtomIds,
                                                   float4 *                    intersectionsNeighbors,
                                                   float &                     computeTime );
} // namespace bcs::sesdf

struct SesdfBenchmark
{
    float circle;
    float intersection;
    float segment;
    float neighborhood;
};

namespace bcs
{
    SesdfBenchmark sesdfBenchmark( ConstSpan<Vec4f> molecule );
} // namespace bcs

#include "sesdf_benchmark.inl"

#endif // SESDF_BENCHMARK_CUH
