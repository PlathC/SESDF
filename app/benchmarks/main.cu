#include <fstream>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

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

// Benchmark tooling
using Task = std::function<void()>;
double cudaTimer( const Task & task )
{
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start );

    task();

    cudaEventRecord( stop );

    cudaEventSynchronize( stop );
    float milliseconds = 0;
    cudaEventElapsedTime( &milliseconds, start, stop );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return milliseconds;
}

std::vector<float> timeBenchmark( const std::function<double( Task )> & timerFunction,
                                  const Task &                          task,
                                  const std::size_t                     iterationNb )
{
    std::vector<float> results {};
    results.resize( iterationNb );
    for ( std::size_t i = 0; i < iterationNb; i++ )
    {
        const float currentTime = timerFunction( task );
        results[ i ]            = currentTime;
    }

    return results;
}

std::vector<float> benchmark( const Task &                                  task,
                              const std::size_t                             iterationNb,
                              const std::function<double( const Task & )> & timerFunction,
                              const bool                                    warmup )
{
    if ( warmup )
        timeBenchmark( timerFunction, task, 100 );

    return timeBenchmark( timerFunction, task, iterationNb );
}

#ifndef BCS_BENCHMARKCBSESDF_PR
#define BCS_BENCHMARKCBSESDF_PR 1.4f
#endif // BCS_BENCHMARKCBSESDF_PR

using ResultType = std::optional<std::vector<float>>;
int main( int argc, char ** argv )
{
    constexpr std::size_t      IterationNb        = 1000;
    constexpr bool             doWarmup           = true;
    constexpr float            ProbeRadius        = BCS_BENCHMARKCBSESDF_PR;
    constexpr uint32_t         MaxNeighborPerAtom = BCS_SESDF_MAXNEIGHBORPERATOM;
    constexpr std::string_view Configuration      = "Ryzen-5-2080";

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
    try
    {
        // CUDA - GL operability requires a proper GL context
        bcs::Context context {};

        // Contour-Buildup benchmarks
        std::array<ResultType, SampleCount> cbResults {};
        for ( std::size_t i : cbCases )
        {
            const bcs::Path     path     = cases[ i ];
            const bcs::Molecule molecule = bcs::load( "samples" / path );
            const bcs::Aabb     aabb     = bcs::getAabb( molecule );

            sizes[ i ] = molecule.size();

            fmt::print( "CB - {} ({} atoms)...\n", cases[ i ], molecule.size() );

            // Grid size as the constant used by Megamol's implementation
            // Reference:
            // https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/protein_cuda/src/MoleculeCBCudaRenderer.cpp#L197
            constexpr uint32_t gridSize = 16;

            bcs::ContourBuildup cb = bcs::ContourBuildup( molecule, aabb, ProbeRadius, gridSize, false );
            cbResults[ i ]         = benchmark( [ &cb ] { cb.build(); }, IterationNb, cudaTimer, doWarmup );
        }

        // Ours - Exterior only benchmarks
        std::array<ResultType, SampleCount> ssesdfResults {};
        for ( std::size_t i : ssesdfCases )
        {
            const bcs::Path     path     = cases[ i ];
            const bcs::Molecule molecule = bcs::load( "samples" / path );
            const bcs::Aabb     aabb     = bcs::getAabb( molecule );

            sizes[ i ] = molecule.size();

            fmt::print( "SSESDF - {} ({} atoms)...\n", cases[ i ], sizes[ i ] );

            bcs::Ssesdf ssesdf = bcs::Ssesdf( molecule, aabb, ProbeRadius, false );
            ssesdfResults[ i ] = benchmark( [ &ssesdf ] { ssesdf.build(); }, IterationNb, cudaTimer, doWarmup );
        }

        // Ours - Complete surface benchmarks
        std::array<ResultType, SampleCount> sesdfResults {};
        for ( std::size_t i : sesdfCases )
        {
            const bcs::Path     path     = cases[ i ];
            const bcs::Molecule molecule = bcs::load( "samples" / path );
            const bcs::Aabb     aabb     = bcs::getAabb( molecule );

            sizes[ i ] = molecule.size();

            fmt::print( "SESDF - {} ({} atoms)...\n", cases[ i ], sizes[ i ] );

            bcs::Sesdf sesdf  = bcs::Sesdf( molecule, aabb, ProbeRadius, false );
            sesdfResults[ i ] = benchmark( [ &sesdf ] { sesdf.build(); }, IterationNb, cudaTimer, doWarmup );
        }

        auto saveResult = []( std::vector<char> &    buffer,
                              const std::string_view pdbId,
                              const std::size_t      atomCount,
                              const std::string_view method,
                              const ResultType &     result )
        {
            if ( !result )
                return;

            const auto & currentResult = *result;
            for ( std::size_t j = 0; j < currentResult.size(); j++ )
            {
                fmt::format_to(
                    std::back_inserter( buffer ), "{};{};{};{};{}\n", pdbId, atomCount, method, j, currentResult[ j ] );
            }
        };

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "Molecule;AtomCount;Method;Iteration;Time\n" );
        for ( std::size_t i = 0; i < cases.size(); i++ )
        {
            const bcs::Path   path = cases[ i ];
            const std::string pdb  = std::filesystem::path( cases[ i ] ).stem().string();
            const std::size_t size = sizes[ i ];

            saveResult( out, pdb, size, "CB", cbResults[ i ] );
            saveResult( out, pdb, size, "SSESDF", ssesdfResults[ i ] );
            saveResult( out, pdb, size, "SESDF", sesdfResults[ i ] );
        }

        // Write results data to a csv file
        auto outputCsv
            = std::ofstream { fmt::format( "./compute-results-{}-ite={}{}-pr={:.1f}-MaxNeighborPerAtom={}.csv",
                                           Configuration,
                                           IterationNb,
                                           doWarmup ? "-warmup" : "",
                                           ProbeRadius,
                                           MaxNeighborPerAtom ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );

        fmt::print( "Done !" );
    }
    catch ( const std::exception & e )
    {
        fmt::print( e.what() );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
