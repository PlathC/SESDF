#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>

#include <bcs/core/context.hpp>
#include <bcs/core/loader.hpp>
#include <bcs/core/molecule.hpp>
#include <cuda.h>
#include <fmt/printf.h>

#include "cb_benchmark.cuh"

int main( int argc, char ** argv )
{
    constexpr std::size_t      WarmupIterationNb = 100;
    constexpr std::size_t      IterationNb       = 1000;
    constexpr std::string_view Configuration     = "Ryzen-5-2080";

    const std::array<std::string, 10> cases {
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
    };

    try
    {
        // CUDA - GL operability requires a proper GL context
        bcs::Context context {};

        using ResultType = std::vector<CBBenchmark>;
        std::vector<ResultType> records;
        records.reserve( cases.size() );

        std::vector<std::size_t> sizes;
        sizes.reserve( cases.size() );
        for ( const bcs::Path sample : cases )
        {
            const bcs::Path     path     = "samples" / sample;
            const bcs::Molecule molecule = bcs::load( path );
            const std::string   pdbId    = path.stem().string();
            fmt::print( "CB Detailed - {} ({} atoms)...\n", pdbId, molecule.size() );

            sizes.emplace_back( molecule.size() );

            for ( std::size_t i = 0; i < WarmupIterationNb; i++ )
                bcs::cbBenchmark( molecule );

            ResultType result {};
            result.reserve( IterationNb );
            for ( std::size_t i = 0; i < IterationNb; i++ )
            {
                const auto times = bcs::cbBenchmark( molecule );
                result.emplace_back( times );
            }

            records.emplace_back( std::move( result ) );
        }

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "Molecule;AtomCount;Iteration;Circle;Intersection;Neighborhood\n" );
        for ( std::size_t i = 0; i < cases.size(); i++ )
        {
            const bcs::Path   path          = cases[ i ];
            const std::string pdb           = path.stem().string();
            const std::size_t size          = sizes[ i ];
            const ResultType  sampleResults = records[ i ];

            for ( std::size_t i = 0; i < sampleResults.size(); i++ )
            {
                const CBBenchmark result = sampleResults[ i ];
                fmt::format_to( std::back_inserter( out ),
                                "{};{};{};{};{};{}\n",
                                pdb,
                                size,
                                i,
                                result.circle,
                                result.intersection,
                                result.neighborhood );
            }
        }

        // Write results data to a csv file
        auto outputCsv      = std::ofstream { fmt::format( "./cb-detailed-results-{}-ite={}{}.csv",
                                                      Configuration,
                                                      IterationNb,
                                                      WarmupIterationNb > 0 ? "-warmup" : "" ) };
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
