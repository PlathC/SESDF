#include <cstdio>
#include <fstream>
#include <numeric>
#include <optional>

#include <cupti.h>
#include <fmt/printf.h>

#include "bcs/contour_buildup/contour_buildup.hpp"
#include "bcs/core/context.hpp"
#include "bcs/core/loader.hpp"
#include "bcs/sesdf/sesdf.hpp"
#include "bcs/ssesdf/ssesdf.hpp"

// Pretty printer for optional
// https://github.com/fmtlib/fmt/issues/1367#issuecomment-1050359900
namespace fmt
{
    template<typename T>
    struct formatter<std::optional<T>> : fmt::formatter<T>
    {
        template<typename FormatContext>
        auto format( const std::optional<T> & opt, FormatContext & ctx )
        {
            if ( opt )
            {
                fmt::formatter<T>::format( *opt, ctx );
                return ctx.out();
            }
            return fmt::format_to( ctx.out(), "-" );
        }
    };
} // namespace fmt

// Based on CUPTI samples
#define CUPTI_CALL( call )                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        CUptiResult _status = call;                                                                                    \
        if ( _status != CUPTI_SUCCESS )                                                                                \
        {                                                                                                              \
            const char * errstr;                                                                                       \
            cuptiGetResultString( _status, &errstr );                                                                  \
            fprintf( stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr ); \
            exit( -1 );                                                                                                \
        }                                                                                                              \
    } while ( 0 )

#define BUF_SIZE ( 32 * 1024 )
#define ALIGN_SIZE ( 8 )
#define ALIGN_BUFFER( buffer, align )                                              \
    ( ( (uintptr_t)( buffer ) & ( (align)-1 ) )                                    \
          ? ( ( buffer ) + ( align ) - ( (uintptr_t)( buffer ) & ( (align)-1 ) ) ) \
          : ( buffer ) )

std::size_t deviceAllocationSize    = 0;
std::size_t deviceMaxAllocationSize = 0;
static void updateMemory( CUpti_Activity * record )
{
    switch ( record->kind )
    {
    case CUPTI_ACTIVITY_KIND_MEMORY2:
    {
        CUpti_ActivityMemory3 * memory = (CUpti_ActivityMemory3 *)(void *)record;
        if ( memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION )
        {
            deviceAllocationSize += memory->bytes;
        }
        else if ( memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE )
        {
            if ( deviceAllocationSize < memory->bytes )
                fmt::print( "Error: More release than allocation." );
            deviceAllocationSize -= memory->bytes;
        }
        break;
    }
    default: break;
    }

    deviceMaxAllocationSize = std::max( deviceAllocationSize, deviceMaxAllocationSize );
}

void CUPTIAPI bufferRequested( uint8_t ** buffer, size_t * size, size_t * maxNumRecords )
{
    uint8_t * bfr = new uint8_t[ BUF_SIZE + ALIGN_SIZE ];

    if ( bfr == NULL )
    {
        fmt::print( "Error: Out of memory.\n" );
        exit( -1 );
    }

    *size          = BUF_SIZE;
    *buffer        = ALIGN_BUFFER( bfr, ALIGN_SIZE );
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted( CUcontext ctx, uint32_t streamId, uint8_t * buffer, size_t size, size_t validSize )
{
    CUptiResult      status;
    CUpti_Activity * record = NULL;

    if ( validSize > 0 )
    {
        do
        {
            status = cuptiActivityGetNextRecord( buffer, validSize, &record );
            if ( status == CUPTI_SUCCESS )
                updateMemory( record );
            else if ( status == CUPTI_ERROR_MAX_LIMIT_REACHED )
                break;
            else
                CUPTI_CALL( status );
        } while ( 1 );

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL( cuptiActivityGetNumDroppedRecords( ctx, streamId, &dropped ) );
        if ( dropped != 0 )
            fmt::print( "Warning: Dropped {} activity records.\n", (unsigned int)dropped );
    }

    delete buffer;
}

void initializeCupti()
{
    CUPTI_CALL( cuptiActivityEnable( CUPTI_ACTIVITY_KIND_MEMORY2 ) );
    CUPTI_CALL( cuptiActivityEnable( CUPTI_ACTIVITY_KIND_MEMORY_POOL ) );

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL( cuptiActivityRegisterCallbacks( bufferRequested, bufferCompleted ) );
}

void reset()
{
    deviceAllocationSize    = 0;
    deviceMaxAllocationSize = 0;
}

int64_t getMaxMemoryUse()
{
    // Force flush any remaining activity buffers before sending the max allocation size
    CUPTI_CALL( cuptiActivityFlushAll( 1 ) );

    return deviceMaxAllocationSize;
}

#ifndef BCS_MEMORYCONSUMPTION_PR
#define BCS_MEMORYCONSUMPTION_PR 1.4f
#endif // BCS_MEMORYCONSUMPTION_PR

int main( int argc, char ** argv )
{
    try
    {
        // CUDA - GL operability requires a proper GL context
        bcs::Context window {};

        constexpr float    ProbeRadius        = BCS_MEMORYCONSUMPTION_PR;
        constexpr uint32_t MaxNeighborPerAtom = BCS_SESDF_MAXNEIGHBORPERATOM;

        const std::array<std::string, 15> cases {
            "1AGA.mmtf", // 126
            "101M.mmtf", // 1,413
            "1VIS.mmtf", // 2,531
            "7SC0.mmtf", // 11,638
            "3EAM.mmtf", // 13,505
            "7DBB.mmtf", // 17,733
            "1A8R.mmtf", // 45,625
            "7O0U.mmtf", // 55,758
            "1AON.mmtf", // 58,870
            "7RGD.mmtf", // 65,008
            "3JC8.mmtf", // 107,640
            "7CGO.mmtf", // 335,722
            "4V4G.mmtf", // 717,805
            "6U42.mmtf", // 1,358,547
            "3J3Q.mmtf"  // 2,440,800
        };
        constexpr std::size_t SampleCount = cases.size();

        std::array<std::size_t, SampleCount> sizes { 0 };

        // Limit Contour-Buildup's test case to avoid overflow
        std::size_t cbCasesBound = 10; // 8GB default limit for NVIDIA RTX 2080
        if ( argc > 1 )
            cbCasesBound = std::atoi( argv[ 1 ] );

        std::size_t ssesdfCasesBound = cases.size();
        if ( argc > 2 )
            ssesdfCasesBound = std::atoi( argv[ 2 ] );

        std::size_t sesdfCasesBound = cases.size();
        if ( argc > 3 )
            sesdfCasesBound = std::atoi( argv[ 3 ] );

        std::vector<std::size_t> cbCases {};
        cbCases.resize( cbCasesBound );
        std::iota( cbCases.begin(), cbCases.end(), 0 );

        std::vector<std::size_t> ssesdfCases {};
        ssesdfCases.resize( ssesdfCasesBound );
        std::iota( ssesdfCases.begin(), ssesdfCases.end(), 0 );

        std::vector<std::size_t> sesdfCases {};
        sesdfCases.resize( sesdfCasesBound );
        std::iota( sesdfCases.begin(), sesdfCases.end(), 0 );

        fmt::print( "CB     - {} / {} cases\n", cbCases.size(), SampleCount );
        fmt::print( "SSESDF - {} / {} cases\n", ssesdfCases.size(), SampleCount );
        fmt::print( "SESDF  - {} / {} cases\n", sesdfCases.size(), SampleCount );

        initializeCupti();

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "Molecule;Method;MB\n" );

        // Grid size as the constant used by Megamol's implementation
        // Reference:
        // https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/protein_cuda/src/MoleculeCBCudaRenderer.cpp#L197
        constexpr uint32_t gridSize = 16;
        for ( std::size_t id : cbCases )
        {
            reset();
            bcs::Path           sample   = cases[ id ];
            const bcs::Molecule molecule = bcs::load( "samples" / sample );
            {
                const bcs::Aabb     aabb    = bcs::getAabb( molecule );
                bcs::ContourBuildup surface = bcs::ContourBuildup( molecule, aabb, ProbeRadius, gridSize, false );
                surface.build();
            }

            const std::string pdb = sample.stem().string();
            fmt::print( "CB - {} ({} atoms)...\n", pdb, molecule.size() );
            fmt::format_to( std::back_inserter( out ), "{};{};{}\n", pdb, "CB", getMaxMemoryUse() / 1e6 );
        }

        for ( std::size_t id : ssesdfCases )
        {
            reset();
            bcs::Path           sample   = cases[ id ];
            const bcs::Molecule molecule = bcs::load( "samples" / sample );
            {
                const bcs::Aabb aabb    = bcs::getAabb( molecule );
                bcs::Ssesdf     surface = bcs::Ssesdf( molecule, aabb, ProbeRadius, false );
                surface.build();
            }

            const std::string pdb = sample.stem().string();
            fmt::print( "SSESDF - {} ({} atoms)...\n", pdb, molecule.size() );
            fmt::format_to( std::back_inserter( out ), "{};{};{}\n", pdb, "SSESDF", getMaxMemoryUse() / 1e6 );
        }

        for ( std::size_t id : sesdfCases )
        {
            reset();
            bcs::Path           sample   = cases[ id ];
            const bcs::Molecule molecule = bcs::load( "samples" / sample );
            {
                const bcs::Aabb aabb    = bcs::getAabb( molecule );
                bcs::Sesdf      surface = bcs::Sesdf( molecule, aabb, ProbeRadius, false );
                surface.build();
            }

            const std::string pdb = sample.stem().string();
            fmt::print( "SESDF - {} ({} atoms)...\n", pdb, molecule.size() );
            fmt::format_to( std::back_inserter( out ), "{};{};{}\n", pdb, "SESDF", getMaxMemoryUse() / 1e6 );
        }

        // Write results data to a csv file
        auto outputCsv      = std::ofstream { fmt::format(
            "./memory-results-pr={:.1f}-MaxNeighborPerAtom={}.csv", ProbeRadius, MaxNeighborPerAtom ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );

        fmt::print( "Done !" );
    }
    catch ( const std::exception & e )
    {
        fmt::print( "Error: {}\n", e.what() );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
