#include <bcs/core/molecule.hpp>
#include <bcs/cuda/execution.cuh>
#include <bcs/cuda/grid.cuh>
#include <bcs/sesdf/data.cuh>
#include <bcs/sesdf/operations.cuh>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/common.hpp>
#include <helper_cuda.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#include "sesdf_benchmark.cuh"

namespace bcs::sesdf
{
    void handleIntersectionSingularitiesBenchmark( AccelerationGridBenchmark & grid,
                                                   const SesdfContext &        sesContext,
                                                   const float4 * const        intersectionsPositions,
                                                   int4 *                      intersectionAtomIds,
                                                   float4 *                    intersectionsNeighbors,
                                                   float &                     computeTime )
    {
        grid.build( sesContext.intersectionWithNeighborNb, intersectionsPositions, computeTime );

        TimingEvent neighborhoodTimer {};
        neighborhoodTimer.start();

        auto [ numBlocks, numThreads ] = KernelConfig::From( sesContext.intersectionWithNeighborNb, 256 );
        findIntersectionsNeighbors<<<numBlocks, numThreads>>>( grid.getConfiguration(),
                                                               sesContext,
                                                               grid.getSortedPosition(),
                                                               grid.getSortedIndices(),
                                                               grid.getCellsStart(),
                                                               grid.getCellsEnd(),
                                                               intersectionsPositions,
                                                               intersectionAtomIds,
                                                               intersectionsNeighbors );
        cudaCheck( "Find Intersections Neighbors failed" );
        neighborhoodTimer.stop();

        computeTime = neighborhoodTimer.getElapsedMs();
    }
} // namespace bcs::sesdf

namespace bcs
{
    SesdfBenchmark sesdfBenchmark( ConstSpan<Vec4f> molecule )
    {
        constexpr uint16_t MaxNeighborPerAtom             = 128;
        constexpr uint16_t MaxMeanIntersectionsPerCircles = 2;
        constexpr uint16_t MaxIntersectionsPerCircles     = 16;
        constexpr uint16_t MaxIntersectionNeighbors       = 32;
        constexpr float    ProbeRadius                    = 1.4f;
        constexpr float    maxVdwRadius                   = 3.48f;
        constexpr bool     useGraphicsBuffers             = true;

        const uint32_t  atomNb = molecule.size;
        const bcs::Aabb aabb   = bcs::getAabb( molecule );

        const glm::vec3 worldOrigin = aabb.min - maxVdwRadius - ProbeRadius;
        const glm::vec3 worldSize   = glm::abs( aabb.max + maxVdwRadius + ProbeRadius - worldOrigin );

        const uint32_t gridSize
            = static_cast<uint32_t>( bcs::nextPowerOfTwoValue( bcs::nextPowerOfTwoExponent( atomNb ) ) );
        const glm::vec3 cellSize = worldSize / static_cast<float>( gridSize );

        bcs::GridInfo gridConfiguration {};
        gridConfiguration.worldOrigin = make_float3( worldOrigin.x, worldOrigin.y, worldOrigin.z );
        gridConfiguration.cellSize    = make_float3( cellSize.x, cellSize.y, cellSize.z );
        gridConfiguration.size        = make_int3( static_cast<int>( gridSize ) );

        // // Pick the device with highest Gflops/s
        cudaDeviceProp deviceProp;
        int            deviceId = gpuGetMaxGflopsDeviceId();
        cudaCheck( cudaSetDevice( deviceId ) );
        cudaCheck( cudaGetDeviceProperties( &deviceProp, deviceId ) );

        int isMemPoolSupported = 0;
        cudaCheck( cudaDeviceGetAttribute( &isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, deviceId ) );
        assert( isMemPoolSupported );

        cudaCheck( cudaSetDevice( deviceId ) );

        uint32_t * hIntersectedCircleNb = nullptr;
        cudaCheck( cudaMallocHost( &hIntersectedCircleNb, sizeof( uint32_t ) ) );
        uint32_t * hIntersectionNb = nullptr;
        cudaCheck( cudaMallocHost( &hIntersectionNb, sizeof( uint32_t ) ) );
        uint32_t * hFullCircleNb = nullptr;
        cudaCheck( cudaMallocHost( &hFullCircleNb, sizeof( uint32_t ) ) );
        uint32_t * hSegmentCount = nullptr;
        cudaCheck( cudaMallocHost( &hSegmentCount, sizeof( uint32_t ) ) );
        uint32_t * hVisibleCircleNb = nullptr;
        cudaCheck( cudaMallocHost( &hVisibleCircleNb, sizeof( uint32_t ) ) );
        uint32_t * dIntersectedCircleNb = nullptr;
        cudaCheck( cudaMalloc( &dIntersectedCircleNb, sizeof( uint32_t ) ) );
        uint32_t * dIntersectionNb = nullptr;
        cudaCheck( cudaMalloc( &dIntersectionNb, sizeof( uint32_t ) ) );
        uint32_t * dFullCircleNb = nullptr;
        cudaCheck( cudaMalloc( &dFullCircleNb, sizeof( uint32_t ) ) );
        uint32_t * dSegmentCount = nullptr;
        cudaCheck( cudaMalloc( &dSegmentCount, sizeof( uint32_t ) ) );

        DeviceBuffer dAtomIndices            = DeviceBuffer::Typed<uint32_t>( atomNb );
        DeviceBuffer dAtomNeighborsCount     = DeviceBuffer::Typed<uint32_t>( atomNb );
        DeviceBuffer dAtomNeighborsIndices   = DeviceBuffer::Typed<uint32_t>( MaxNeighborPerAtom * atomNb );
        DeviceBuffer dCircleVisibilityStatus = DeviceBuffer::Typed<uint8_t>( MaxNeighborPerAtom * atomNb + 1 );
        DeviceBuffer dGlobalToTrimmedId      = DeviceBuffer::Typed<uint32_t>( MaxNeighborPerAtom * atomNb + 1 );

        cudaMemPool_t memPool;
        cudaCheck( cudaDeviceGetDefaultMemPool( &memPool, deviceId ) );

        constexpr uint64_t thresholdVal = std::numeric_limits<uint64_t>::max();
        cudaCheck( cudaMemPoolSetAttribute( memPool, cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal ) );
        AccelerationGridBenchmark accelerationGrid = AccelerationGridBenchmark( gridConfiguration );

        uint32_t intersectionNb = 0;

        ResultBuffer dAtoms = ResultBuffer::Typed<float4>( atomNb );
        ResultBuffer dIntersections;
        ResultBuffer dSegments;
        ResultBuffer dConvexPatches = ResultBuffer::Typed<uint2>( atomNb );
        ResultBuffer dFCircleAndSectors;

        float circleComputeTime       = 0.f;
        float intersectionComputeTime = 0.f;
        float segmentComputeTime      = 0.f;
        float neighborhoodComputeTime = 0.f;
        {
            sesdf::SesdfContext sesContext {};
            sesContext.probeRadius                = ProbeRadius;
            sesContext.atomNb                     = atomNb;
            sesContext.atoms                      = dAtoms.get<float4>();
            sesContext.sortedToInitialIndices     = dAtomIndices.get<uint32_t>();
            sesContext.neighborNb                 = dAtomNeighborsCount.get<uint32_t>();
            sesContext.neighborIds                = dAtomNeighborsIndices.get<uint32_t>();
            sesContext.visibilityStatus           = dCircleVisibilityStatus.get<uint8_t>();
            sesContext.maxNeighborPerAtom         = MaxNeighborPerAtom;
            sesContext.maxIntersectionsPerCircles = MaxMeanIntersectionsPerCircles;
            sesContext.maxIntersectionNeighbors   = MaxIntersectionNeighbors;
            sesContext.globalToTrimmedId          = dGlobalToTrimmedId.get<uint32_t>();

            sesContext.dIntersectedCircleNb = dIntersectedCircleNb;
            cudaMemset( sesContext.dIntersectedCircleNb, 0, sizeof( uint32_t ) );
            sesContext.hIntersectedCircleNb  = hIntersectedCircleNb;
            *sesContext.hIntersectedCircleNb = 0;

            sesContext.dIntersectionNb = dIntersectionNb;
            cudaMemset( sesContext.dIntersectionNb, 0, sizeof( uint32_t ) );
            sesContext.hIntersectionNb  = hIntersectionNb;
            *sesContext.hIntersectionNb = 0;

            sesContext.dFullCircleNb = dFullCircleNb;
            cudaMemset( sesContext.dFullCircleNb, 0, sizeof( uint32_t ) );
            sesContext.hFullCircleNb  = hFullCircleNb;
            *sesContext.hFullCircleNb = 0;

            sesContext.dSegmentCount = dSegmentCount;
            cudaMemset( sesContext.dSegmentCount, 0, sizeof( uint32_t ) );
            sesContext.hSegmentCount  = hSegmentCount;
            *sesContext.hSegmentCount = 0;

            sesContext.hVisibleCircleNb  = hVisibleCircleNb;
            *sesContext.hVisibleCircleNb = 0;

            // #1: CPU => GPU
            mmemcpy<MemcpyType::HostToDevice>(
                dAtoms.get<float4>(), reinterpret_cast<const float4 *>( molecule.ptr ), atomNb );

            // #2: Find Circles
            auto [ fCircleAndSectors, trimmedToGlobalId ] = sesdf::findCirclesBenchmark<MaxNeighborPerAtom>(
                accelerationGrid, sesContext, dConvexPatches.get<uint2>(), useGraphicsBuffers, circleComputeTime );

            sesContext.trimmedToGlobalId = trimmedToGlobalId.get<uint32_t>();
            dFCircleAndSectors           = std::move( fCircleAndSectors );

            if ( *sesContext.hIntersectedCircleNb > 0 )
            {
                // #3: Find intersections
                auto circlesIntersectionsNb = DeviceBuffer::Typed<uint32_t>( *sesContext.hIntersectedCircleNb, true );
                sesContext.circlesIntersectionNb = circlesIntersectionsNb.get<uint32_t>();

                auto [ ddIntersections, startandIdList ] = sesdf::findIntersectionsBenchmark<MaxNeighborPerAtom>(
                    sesContext, useGraphicsBuffers, intersectionComputeTime );

                intersectionNb = *sesContext.hIntersectionNb;
                dIntersections = std::move( ddIntersections );

                sesContext.circlesIntersectionStartId = startandIdList.get<uint32_t>();
                sesContext.circlesIntersectionIds = startandIdList.get<uint32_t>() + *sesContext.hIntersectedCircleNb;

                // #4: Build P_t
                auto segmentsData = sesdf::buildConicPatchesBenchmark<MaxIntersectionsPerCircles>(
                    sesContext,
                    dIntersections.get<const float4>( intersectionNb * sizeof( int4 ) ),
                    dIntersections.get<const int4>(),
                    useGraphicsBuffers,
                    segmentComputeTime );

                dSegments = std::move( segmentsData );

                // #5: Build P_- neighbors
                sesdf::handleIntersectionSingularitiesBenchmark(
                    accelerationGrid,
                    sesContext,
                    dIntersections.get<const float4>( intersectionNb * sizeof( int4 ) ),
                    dIntersections.get<int4>(),
                    dIntersections.get<float4>( intersectionNb * ( sizeof( int4 ) + sizeof( float4 ) ) ),
                    neighborhoodComputeTime );

                dIntersections.unmap();
                dSegments.unmap();
                dConvexPatches.unmap();
                dFCircleAndSectors.unmap();
            }
        }
        cudaCheck( cudaFreeHost( hIntersectedCircleNb ) );
        cudaCheck( cudaFreeHost( hIntersectionNb ) );
        cudaCheck( cudaFreeHost( hFullCircleNb ) );
        cudaCheck( cudaFreeHost( hSegmentCount ) );
        cudaCheck( cudaFreeHost( hVisibleCircleNb ) );
        cudaCheck( cudaFree( dIntersectedCircleNb ) );
        cudaCheck( cudaFree( dIntersectionNb ) );
        cudaCheck( cudaFree( dFullCircleNb ) );
        cudaCheck( cudaFree( dSegmentCount ) );

        return SesdfBenchmark {
            circleComputeTime, intersectionComputeTime, segmentComputeTime, neighborhoodComputeTime
        };
    }
} // namespace bcs
