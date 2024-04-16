#include <bcs/sesdf/graphics.hpp>
#include <bcs/sesdf/operations.cuh>
#include <cooperative_groups.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "sesdf_benchmark.cuh"

namespace bcs::sesdf
{
    template<uint32_t MaxNeighborPerAtom>
    std::pair<ResultBuffer, DeviceBuffer> findCirclesBenchmark( AccelerationGridBenchmark & grid,
                                                                bcs::sesdf::SesdfContext &  sesContext,
                                                                uint2 *                     convexPatches,
                                                                bool                        graphics,
                                                                float &                     computeTime )
    {
        grid.build( sesContext.atomNb, sesContext.atoms, computeTime );

        TimingEvent circleTimer {};
        circleTimer.start();
        {
            copy( sesContext.atoms, grid.getSortedPosition(), sesContext.atomNb );
            copy( sesContext.sortedToInitialIndices, grid.getSortedIndices(), sesContext.atomNb );

            auto [ numBlocks, numThreads ] = KernelConfig::From( sesContext.atomNb, sesContext.maxNeighborPerAtom );
            findCirclesBetweenAtoms<<<numBlocks, numThreads>>>(
                grid.getConfiguration(), grid.getCellsStart(), grid.getCellsEnd(), sesContext );
            cudaCheck( "Circles evaluation failed" );
        }
        circleTimer.stop();
        computeTime += circleTimer.getElapsedMs();

        // Circles visibility status
        DeviceBuffer temp { sesContext.getMaximumCircleNb() * sizeof( uint2 )
                                + ( sesContext.getMaximumCircleNb() + 1 ) * sizeof( uint32_t ),
                            true };

        uint32_t * mask
            = reinterpret_cast<uint32_t *>( temp.get<uint8_t>() + sesContext.getMaximumCircleNb() * sizeof( uint2 ) );
        uint2 * tempFullCircles = temp.get<uint2>();

        TimingEvent visibilityTimer {};
        visibilityTimer.start();
        {
            computeCirclesVisibilityStatus<<<sesContext.atomNb,
                                             sesContext.maxNeighborPerAtom,
                                             sesContext.maxNeighborPerAtom * sizeof( float4 )>>>(
                sesContext, tempFullCircles, mask );
            cudaCheck( "Circles visibility evaluation failed" );

            prefixSumCount( mask,
                            sesContext.globalToTrimmedId,
                            sesContext.hIntersectedCircleNb,
                            sesContext.getMaximumCircleNb() + 1 );
        }
        visibilityTimer.stop();
        computeTime += visibilityTimer.getElapsedMs();

        auto trimmedToGlobalId = DeviceBuffer::Typed<uint32_t>( *sesContext.hIntersectedCircleNb );

        TimingEvent transformationTimer {};
        transformationTimer.start();
        {
            thrust::copy_if( thrust::device,
                             thrust::make_counting_iterator<uint32_t>( 0 ),
                             thrust::make_counting_iterator<uint32_t>( sesContext.getMaximumCircleNb() ),
                             mask,
                             trimmedToGlobalId.get<uint32_t>(),
                             thrust::identity<uint32_t>() );

            // Convex patches
            thrust::transform_exclusive_scan(
                thrust::device,
                sesContext.visibilityStatus,
                sesContext.visibilityStatus + sesContext.getMaximumCircleNb() + 1,
                mask,
                [] __device__( uint8_t x ) { return uint32_t( x != 0 ); },
                0,
                thrust::plus<uint32_t>() );
            mmemcpy<MemcpyType::DeviceToHost>( sesContext.hVisibleCircleNb, mask + sesContext.getMaximumCircleNb(), 1 );
            mmemcpy<MemcpyType::DeviceToHost>( sesContext.hFullCircleNb, sesContext.dFullCircleNb, 1 );
        }
        transformationTimer.stop();
        computeTime += transformationTimer.getElapsedMs();

        const uint32_t    visibleCirclesNb = *sesContext.hVisibleCircleNb;
        const std::size_t alignment        = ( *sesContext.hFullCircleNb * sizeof( uint2 ) ) % sizeof( float4 );
        ResultBuffer      fCirclesAndSectors { *sesContext.hFullCircleNb * sizeof( uint2 ) + alignment
                                              + visibleCirclesNb * sizeof( float4 ),
                                          false,
                                          graphics };

        TimingEvent convexPatchTimer {};
        convexPatchTimer.start();
        {
            if ( visibleCirclesNb > 0 )
            {
                buildConvexPatchesSectors<MaxNeighborPerAtom><<<sesContext.atomNb, sesContext.maxNeighborPerAtom>>>(
                    sesContext,
                    mask,
                    convexPatches,
                    fCirclesAndSectors.get<float4>( *sesContext.hFullCircleNb * sizeof( uint2 ) + alignment ) );
                cudaCheck( "Convex patch construction failed" );
            }

            if ( *sesContext.hFullCircleNb > 0 )
                copy( fCirclesAndSectors.get<uint2>(), tempFullCircles, *sesContext.hFullCircleNb );
        }
        convexPatchTimer.stop();
        computeTime += convexPatchTimer.getElapsedMs();

        return std::make_pair<ResultBuffer, DeviceBuffer>( std::move( fCirclesAndSectors ),
                                                           std::move( trimmedToGlobalId ) );
    }

    template<uint32_t MaxNeighborPerAtom>
    std::pair<ResultBuffer, DeviceBuffer> findIntersectionsBenchmark( SesdfContext & sesContext,
                                                                      bool           graphics,
                                                                      float &        computeTime )
    {
        ResultBuffer intersectionBuffer;

        TimingEvent intersectionTimer {};
        {
            auto tempIntersectionsIds = DeviceBuffer::Typed<int4>( ( *sesContext.hIntersectedCircleNb )
                                                                   * sesContext.maxIntersectionsPerCircles );

            intersectionTimer.start();
            findIntersectionsBetweenCircles<MaxNeighborPerAtom>
                <<<sesContext.atomNb, sesContext.maxNeighborPerAtom>>>( sesContext, tempIntersectionsIds.get<int4>() );
            cudaCheck( "Intersection construction failed" );

            mmemcpy<MemcpyType::DeviceToHost>( sesContext.hIntersectionNb, sesContext.dIntersectionNb, 1 );

            if ( *sesContext.hIntersectionNb
                 > ( *sesContext.hIntersectedCircleNb ) * sesContext.maxIntersectionsPerCircles )
            {
                fmt::print( "Error: More intersection than expected",
                            *sesContext.hIntersectionNb,
                            ( *sesContext.hIntersectedCircleNb ) * sesContext.maxIntersectionsPerCircles );
            }

            // Place with-neighbors intersections at the begin of the array
            int4 * intersectionIds = tempIntersectionsIds.get<int4>();
            thrust::sort( thrust::device,
                          intersectionIds,
                          intersectionIds + *sesContext.hIntersectionNb,
                          HasNeighborsComparator() );
            sesContext.intersectionWithNeighborNb = thrust::count_if(
                thrust::device, intersectionIds, intersectionIds + *sesContext.hIntersectionNb, HasNeighbors() );
            intersectionBuffer = ResultBuffer(
                ( *sesContext.hIntersectionNb ) * ( sizeof( int4 ) + sizeof( float4 ) )
                    + sesContext.intersectionWithNeighborNb * sizeof( float4 ) * sesContext.maxIntersectionNeighbors,
                false,
                graphics );
            copy( intersectionBuffer.get<int4>(), tempIntersectionsIds.get<int4>(), *sesContext.hIntersectionNb );
        }
        intersectionTimer.stop();
        computeTime += intersectionTimer.getElapsedMs();

        const uint32_t circleIntersectionNb = ( *sesContext.hIntersectionNb ) * 3;
        auto           temp = DeviceBuffer::Typed<uint32_t>( *sesContext.hIntersectedCircleNb + circleIntersectionNb );
        uint32_t *     startIds = temp.get<uint32_t>();
        uint32_t *     ids      = temp.get<uint32_t>() + *sesContext.hIntersectedCircleNb;
        {
            auto circleIntersectionStencil = DeviceBuffer::Typed<uint32_t>( *sesContext.hIntersectedCircleNb, true );

            TimingEvent fillIntersectionTimer {};
            fillIntersectionTimer.start();
            {
                thrust::transform_exclusive_scan(
                    thrust::device,
                    sesContext.circlesIntersectionNb,
                    sesContext.circlesIntersectionNb + ( *sesContext.hIntersectedCircleNb ),
                    startIds,
                    [] __device__( uint8_t c ) { return static_cast<uint32_t>( c ); },
                    0,
                    thrust::plus<uint32_t>() );

                auto [ blockNb, threadNb ] = KernelConfig::From( *sesContext.hIntersectionNb, 256 );
                fillIntersections<<<blockNb, threadNb>>>(
                    sesContext,
                    startIds,
                    intersectionBuffer.get<int4>(),
                    circleIntersectionStencil.get<uint32_t>(),
                    ids,
                    reinterpret_cast<float4 *>( intersectionBuffer.get<uint8_t>()
                                                + sizeof( int4 ) * ( *sesContext.hIntersectionNb ) ) );
                cudaCheck( "Intersection fill failed" );
            }
            fillIntersectionTimer.stop();
            computeTime += fillIntersectionTimer.getElapsedMs();
        }

        return std::make_pair<ResultBuffer, DeviceBuffer>( std::move( intersectionBuffer ), std::move( temp ) );
    }

    template<uint8_t MaxIntersectionsPerCircle>
    ResultBuffer buildConicPatchesBenchmark( SesdfContext &       sesContext,
                                             const float4 * const intersectionsPositions,
                                             const int4 * const   intersectionAtomIds,
                                             bool                 graphics,
                                             float &              computeTime )
    {
        // Extract circles with segment's ids and segments number from circles status
        const std::size_t buffer1Size = *sesContext.hIntersectedCircleNb + 1;
        auto              temp = DeviceBuffer::Typed<uint32_t>( buffer1Size + *sesContext.hIntersectedCircleNb, true );
        uint32_t *        segmentIds            = temp.get<uint32_t>();
        uint32_t *        circleWithSegmentsIds = temp.get<uint32_t>() + buffer1Size;

        TimingEvent transformTimer {};
        transformTimer.start();
        uint32_t circleWithSegmentNb;
        {
            thrust::transform( thrust::device,
                               sesContext.circlesIntersectionNb,
                               sesContext.circlesIntersectionNb + ( *sesContext.hIntersectedCircleNb ),
                               segmentIds,
                               [] __device__( uint8_t intersectionNb ) { return intersectionNb / 2u; } );

            uint32_t * lastIndex
                = thrust::copy_if( thrust::device,
                                   thrust::make_counting_iterator<uint32_t>( 0 ),
                                   thrust::make_counting_iterator<uint32_t>( *sesContext.hIntersectedCircleNb ),
                                   segmentIds,
                                   circleWithSegmentsIds,
                                   [] __device__( uint32_t x ) { return static_cast<uint32_t>( x != 0 ); } );

            circleWithSegmentNb = std::distance( circleWithSegmentsIds, lastIndex );

            // The sum provides indices for direct saving during segments extraction
            prefixSumCount( segmentIds, segmentIds, sesContext.hSegmentCount, buffer1Size );
        }
        transformTimer.stop();
        computeTime += transformTimer.getElapsedMs();

        ResultBuffer segmentsDataBuffer;
        if ( *sesContext.hSegmentCount > 0 )
        {
            segmentsDataBuffer = ResultBuffer::Typed<uint4>( *sesContext.hSegmentCount, false, graphics );

            TimingEvent segmentTimer {};
            segmentTimer.start();

            auto [ numBlocks, numThreads ] = KernelConfig::From( circleWithSegmentNb, 256 );

            constexpr uint32_t WarpThreadNb = 16;
            if ( numThreads.x % WarpThreadNb != 0 )
                numThreads.x = circleWithSegmentNb + WarpThreadNb - circleWithSegmentNb % WarpThreadNb;

            buildSegments<256, MaxIntersectionsPerCircle><<<numBlocks, numThreads>>>( sesContext,
                                                                                      circleWithSegmentNb,
                                                                                      circleWithSegmentsIds,
                                                                                      intersectionsPositions,
                                                                                      intersectionAtomIds,
                                                                                      segmentIds,
                                                                                      segmentsDataBuffer.get<uint4>() );
            cudaCheck( "Segments construction failed" );
            segmentTimer.stop();
            computeTime += segmentTimer.getElapsedMs();
        }

        return std::move( segmentsDataBuffer );
    }
} // namespace bcs::sesdf