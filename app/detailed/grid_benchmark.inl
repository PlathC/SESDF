#include "bcs/cuda/grid.cuh"

namespace bcs
{
    inline const GridInfo & AccelerationGridBenchmark::getConfiguration() const { return m_configuration; }
    inline const uint32_t * AccelerationGridBenchmark::getCellsStart() const { return m_dCellsStart.get<uint32_t>(); }
    inline const uint32_t * AccelerationGridBenchmark::getCellsEnd() const { return m_dCellsEnd.get<uint32_t>(); }
    inline const uint32_t * AccelerationGridBenchmark::getSortedIndices() const { return m_dIndices.get<uint32_t>(); }
    inline const uint32_t * AccelerationGridBenchmark::getSortedIndices() { return m_dIndices.get<uint32_t>(); }
    inline const float4 *   AccelerationGridBenchmark::getSortedPosition() const
    {
        return m_dSortedPositions.get<float4>();
    }
    inline AccelerationGridBenchmark::Cells AccelerationGridBenchmark::getCells() const
    {
        return { getCellsStart(), getCellsEnd() };
    }
} // namespace bcs
