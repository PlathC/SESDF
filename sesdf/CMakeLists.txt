project(BENCHMARKS_CB_SESDF)

add_subdirectory(extern)

if (CMAKE_COMPILER_IS_GNUCC)
    set(
            BCS_COMPILATION_FLAGS
            -Wall
            -Wfatal-errors
    )
elseif (MSVC)
    set(BCS_COMPILATION_FLAGS
            /permissive
            /fp:fast
    )

    SET(
            BCS_COMPILATION_DEFINITION
            -D_CRT_SECURE_NO_WARNINGS
            -DNOMINMAX
    )
endif ()

set(BCS_COMPILATION_FLAGS ${BCS_COMPILATION_FLAGS} PARENT_SCOPE)
set(BCS_COMPILATION_DEFINITION ${BCS_COMPILATION_DEFINITION} PARENT_SCOPE)

# Get all sources.
set(HEADERS
        "include/bcs/core/context.hpp"
        "include/bcs/core/loader.hpp"
        "include/bcs/core/math.hpp"
        "include/bcs/core/meta.hpp"
        "include/bcs/core/molecule.hpp"
        "include/bcs/core/type.hpp"

        "include/bcs/cuda/circles.cuh"
        "include/bcs/cuda/execution.cuh"
        "include/bcs/cuda/grid.cuh"
        "include/bcs/cuda/math.cuh"
        "include/bcs/cuda/memory.cuh"
        "include/bcs/cuda/setup.cuh"

        "include/bcs/sesdf/data.cuh"
        "include/bcs/sesdf/graphics.hpp"
        "include/bcs/sesdf/operations.cuh"
        "include/bcs/sesdf/sesdf.hpp"

        "include/bcs/ssesdf/data.cuh"
        "include/bcs/ssesdf/graphics.hpp"
        "include/bcs/ssesdf/operations.cuh"
        "include/bcs/ssesdf/ssesdf.hpp"
)

set(SOURCES
        "src/core/context.cpp"
        "src/core/loader.cpp"
        "src/core/molecule.cpp"
        "src/cuda/grid.cu"
        "src/cuda/memory.cu"
        "src/sesdf/operations.cu"
        "src/sesdf/sesdf.cu"
        "src/ssesdf/operations.cu"
        "src/ssesdf/ssesdf.cu"
)

if (SESDF_WITH_ANALYSIS)
    set(SOURCES
            ${SOURCES}
            "src/contour_buildup/contour_buildup.cu"
    )
endif ()

add_library(sesdf ${SOURCES} ${HEADERS})
target_compile_features(sesdf PRIVATE cxx_std_17)
get_compute_architecture(sesdf_compute_arch)
set_target_properties(sesdf PROPERTIES CUDA_ARCHITECTURES ${sesdf_compute_arch})
set_target_properties(sesdf PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(sesdf PRIVATE ${BCS_COMPILATION_DEFINITION})
target_compile_options(sesdf PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BCS_COMPILATION_FLAGS}>)
target_include_directories(sesdf PUBLIC "include/" ${BCS_EXTERN_INCLUDES})
target_link_libraries(sesdf PUBLIC ${BCS_EXTERN_LIBRARIES})
target_compile_options(sesdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --extended-lambda
        -Xcudafe
        --diag_suppress=esa_on_defaulted_function_ignored
        >
)

add_library(sesdf288 ${SOURCES} ${HEADERS})
target_compile_features(sesdf288 PRIVATE cxx_std_17)
get_compute_architecture(sesdf288_compute_arch)
set_target_properties(sesdf288 PROPERTIES CUDA_ARCHITECTURES ${sesdf288_compute_arch})
set_target_properties(sesdf288 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(sesdf288 PRIVATE ${BCS_COMPILATION_DEFINITION} BCS_SESDF_MAXNEIGHBORPERATOM=288 BCS_SSESDF_MAXNEIGHBORPERATOM=288)
target_compile_options(sesdf288 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BCS_COMPILATION_FLAGS}>)
target_include_directories(sesdf288 PUBLIC "include/" ${BCS_EXTERN_INCLUDES})
target_link_libraries(sesdf288 PUBLIC ${BCS_EXTERN_LIBRARIES})
target_compile_options(sesdf288 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --extended-lambda
        -Xcudafe
        --diag_suppress=esa_on_defaulted_function_ignored
        >
)
