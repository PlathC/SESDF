#include <bcs/cuda/math.cuh>
#include <bcs/cuda/setup.cuh>
#include <device_launch_parameters.h>
#include <thrust/async/sort.h>
#include <thrust/sort.h>
#include <vector_functions.h>

#include "grid_benchmark.cuh"
#include "timer.cuh"

namespace bcs
{
    AccelerationGridBenchmark::AccelerationGridBenchmark( GridInfo configuration, cudaStream_t stream ) :
        m_configuration( configuration ),
        m_cellNb( configuration.size.x * configuration.size.y * configuration.size.z ),
        m_dCellsStart( DeviceBuffer::Typed<uint32_t>( m_cellNb, stream ) ),
        m_dCellsEnd( DeviceBuffer::Typed<uint32_t>( m_cellNb, stream ) )
    {
    }

    void AccelerationGridBenchmark::build( const uint32_t       elementNb,
                                           const float4 * const positions,
                                           float &              computeTime )
    {
        cudaMemset( m_dCellsStart.get<uint32_t>(), EmptyGridCellValue, sizeof( uint32_t ) * m_cellNb );

        // Re-allocatation only if needed
        if ( !m_dHashes || m_dHashes.size<uint32_t>() < elementNb )
            m_dHashes = DeviceBuffer::Typed<uint32_t>( elementNb );

        if ( !m_dIndices || m_dIndices.size<uint32_t>() < elementNb )
            m_dIndices = DeviceBuffer::Typed<uint32_t>( elementNb );

        if ( !m_dSortedPositions || m_dSortedPositions.size<float4>() < elementNb )
            m_dSortedPositions = DeviceBuffer::Typed<float4>( elementNb );

        auto [ numBlocks, numThreads ] = KernelConfig::From( elementNb, 256 );

        uint32_t * hashes  = m_dHashes.get<uint32_t>();
        uint32_t * indices = m_dIndices.get<uint32_t>();

        TimingEvent gridTimer {};
        gridTimer.start();

        {
            computeHashes<<<numBlocks, numThreads>>>( m_configuration, elementNb, positions, hashes, indices );
            cudaCheck( "Hashes computation failed" );

            thrust::sort_by_key( thrust::device, hashes, hashes + elementNb, indices );

            const uint32_t sharedMemorySize = sizeof( uint32_t ) * ( numThreads.x + 1 );
            buildAccelerationGrid<<<numBlocks, numThreads, sharedMemorySize>>>( elementNb,
                                                                                hashes,
                                                                                indices,
                                                                                positions,
                                                                                m_dCellsStart.get<uint32_t>(),
                                                                                m_dCellsEnd.get<uint32_t>(),
                                                                                m_dSortedPositions.get<float4>() );
            cudaCheck( "Acceleration grid construction failed" );
        }

        gridTimer.stop();
        computeTime += gridTimer.getElapsedMs();
    }

} // namespace bcs
