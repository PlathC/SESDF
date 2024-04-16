#include <array>
#include <fstream>
#include <numeric>
#include <optional>

#include <fmt/printf.h>

#include "bcs/core/context.hpp"
#include "bcs/core/loader.hpp"
#include "bcs/sesdf/graphics.hpp"
#include "bcs/sesdf/sesdf.hpp"

std::string getCircleData( bcs::Sesdf & sesdf )
{
    bcs::sesdf::SesdfData data = sesdf.getData();

    auto &     dAtomNeighborsCount = sesdf.getDAtomNeighborsCount();
    const auto neighborCount       = dAtomNeighborsCount.toHost<uint32_t>();

    auto &     dAtomNeighborsIndices = sesdf.getDAtomNeighborsIndices();
    const auto neighborIndices       = dAtomNeighborsIndices.toHost<uint32_t>();

    auto & dCircleVisibilityStatus = sesdf.getDCircleVisibilityStatus();
    auto   circlesStatus           = dCircleVisibilityStatus.toHost<uint8_t>();

    std::vector<uint4> segments {};
    segments.resize( data.segmentPatchNb );
    bcs::cudaCheck(
        cudaMemcpy( segments.data(), data.segmentPatches, sizeof( uint4 ) * segments.size(), cudaMemcpyDeviceToHost ) );

    // Status are initially equal to 0 if the circle is buried, 1 if intersected and 2 if complete.
    // We thus set to 3 those who created a segments to separate contributing and non-contributing intersecting circles
    const uint32_t maxNeigborsPerAtom = sesdf.getMaxNeighborPerAtom();
    for ( const uint4 segment : segments )
    {
        // Find circle index by searching for local j index in i's neighbors
        const uint32_t          neighborNb = neighborCount[ segment.x ];
        std::optional<uint32_t> localJ;
        for ( uint32_t i = 0; !localJ && i < neighborNb; i++ )
        {
            if ( neighborIndices[ segment.x * maxNeigborsPerAtom + i ] == segment.y )
                localJ = i;
        }

        if ( !localJ )
        {
            fmt::print( "Error: Unknown neighbor\n" );
            continue;
        }

        // Set current circle's status to intersected with visible intersection
        auto & current = circlesStatus[ segment.x * maxNeigborsPerAtom + *localJ ];
        if ( current != 1 && current != 3 )
            fmt::print( "Found non intersecting circle with status != 1 && != 3 and == {}", current );
        current = 3;
    }

    uint32_t full                 = 0;
    uint32_t buried               = 0;
    uint32_t withoutIntersections = 0;
    uint32_t withIntersections    = 0;
    uint32_t total                = 0;
    for ( uint32_t i = 0; i < data.atomNb; i++ )
    {
        const uint32_t neighborNb = neighborCount[ i ];
        for ( uint32_t localJ = 0; localJ < neighborNb; localJ++ )
        {
            const uint8_t  status = circlesStatus[ i * maxNeigborsPerAtom + localJ ];
            const uint32_t j      = neighborIndices[ i * maxNeigborsPerAtom + localJ ];

            if ( j < i ) // Do not count doubled
                continue;

            total++;
            switch ( status )
            {
            case 0: buried++; break;
            case 1: withoutIntersections++; break;
            case 2: full++; break;
            case 3: withIntersections++; break;
            default: fmt::print( "Error: Unknown circle status.\n" );
            }
        }
    }

    return fmt::format( "{};{};{};{};{}", buried, full, withoutIntersections, withIntersections, total );
}

int main( int /* argc */, char ** /* argv */ )
{
    constexpr float                 ProbeRadius = 1.4f;
    const std::array<bcs::Path, 15> samples {
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

    bcs::Context context {};

    std::string circleStudyCsv
        = "Molecule;AtomCount;Buried;Full;IntersectedWithoutIntersections;IntersectedWithIntersections;Total\n";
    for ( const auto & entry : samples )
    {
        const std::string pdb      = entry.stem().string();
        bcs::Molecule     molecule = bcs::load( "samples" / entry );
        fmt::print( "{} ({} atoms)...\n", pdb, molecule.size() );

        bcs::Sesdf sesdf { molecule, bcs::getAabb( molecule ), ProbeRadius, true };
        circleStudyCsv += fmt::format( "{};{};{}\n", pdb, molecule.size(), getCircleData( sesdf ) );
    }

    // Write results data to a csv file
    auto outputCsv      = std::ofstream { "./circles-results.csv" };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( circleStudyCsv.begin(), circleStudyCsv.end(), outputIterator );

    fmt::print( "Done !" );

    return EXIT_SUCCESS;
}
