#ifndef CONTOURBUILDUP_BENCHMARK_CUH
#define CONTOURBUILDUP_BENCHMARK_CUH

#include <ContourBuildup/particles_kernel.cuh>
#include "timer.cuh"

struct CBBenchmark
{
    float circle;
    float intersection;
    float neighborhood;
};

namespace bcs
{
    CBBenchmark cbBenchmark( ConstSpan<Vec4f> molecule );
} // namespace bcs

#endif // CONTOURBUILDUP_BENCHMARK_CUH
