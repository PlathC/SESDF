#ifndef BCS_CUDA_GRIDBENCHMARK_CUH
#define BCS_CUDA_GRIDBENCHMARK_CUH

#include <bcs/cuda/memory.cuh>
#include <bcs/cuda/grid.cuh>

namespace bcs
{
    class AccelerationGridBenchmark
    {
      public:
        static constexpr uint32_t EmptyGridCellValue = 0xffffffff;

        AccelerationGridBenchmark() = default;
        AccelerationGridBenchmark( GridInfo configuration, cudaStream_t stream = 0 );

        AccelerationGridBenchmark( const AccelerationGridBenchmark & ) = delete;
        AccelerationGridBenchmark & operator=( const AccelerationGridBenchmark & ) = delete;

        AccelerationGridBenchmark( AccelerationGridBenchmark && ) = default;
        AccelerationGridBenchmark & operator=( AccelerationGridBenchmark && ) = default;

        ~AccelerationGridBenchmark() = default;

        void build( const uint32_t elementNb, const float4 * const positions, float & computeTime );

        inline const GridInfo & getConfiguration() const;
        inline const uint32_t * getCellsStart() const;
        inline const uint32_t * getCellsEnd() const;
        inline const uint32_t * getSortedIndices() const;
        inline const uint32_t * getSortedIndices();
        inline const float4 *   getSortedPosition() const;

        struct Cells
        {
            const uint32_t * starts;
            const uint32_t * ends;
        };
        inline Cells getCells() const;

      private:
        GridInfo m_configuration {};
        uint32_t m_cellNb = 0;

        DeviceBuffer m_dSortedPositions;
        DeviceBuffer m_dHashes;
        DeviceBuffer m_dIndices;
        DeviceBuffer m_dCellsStart;
        DeviceBuffer m_dCellsEnd;
    };
} // namespace bcs

#include "grid_benchmark.inl"

#endif // BCS_CUDA_GRIDBENCHMARK_CUH
