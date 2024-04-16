add_executable(BCSBenchmarkCBSESDF "main.cu")
target_compile_features(BCSBenchmarkCBSESDF PRIVATE cxx_std_17)
get_compute_architecture(bcs_benchmarkcbsesdf_compute_arch)
set_target_properties(BCSBenchmarkCBSESDF PROPERTIES CUDA_ARCHITECTURES ${bcs_benchmarkcbsesdf_compute_arch})
set_target_properties(BCSBenchmarkCBSESDF PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSBenchmarkCBSESDF PRIVATE ${BCS_COMPILATION_DEFINITION} 
    BCS_BENCHMARKCBSESDF_PR=1.4f 
    BCS_SESDF_MAXNEIGHBORPERATOM=128 
    BCS_SSESDF_MAXNEIGHBORPERATOM=128
)
target_link_libraries(BCSBenchmarkCBSESDF PRIVATE sesdf)
target_compile_options(BCSBenchmarkCBSESDF PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    --relocatable-device-code=true
    --extended-lambda
    -Xcudafe 
    --diag_suppress=esa_on_defaulted_function_ignored
    >
)

add_dependencies(BCSBenchmarkCBSESDF BCS_SAMPLES)

add_executable(BCSBenchmarkCBSESDF2-4 "main.cu")
target_compile_features(BCSBenchmarkCBSESDF2-4 PRIVATE cxx_std_17)

get_compute_architecture(bcs_benchmarkcbsesdf24_compute_arch)
set_target_properties(BCSBenchmarkCBSESDF2-4 PROPERTIES CUDA_ARCHITECTURES ${bcs_benchmarkcbsesdf24_compute_arch})
set_target_properties(BCSBenchmarkCBSESDF2-4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSBenchmarkCBSESDF2-4 PRIVATE ${BCS_COMPILATION_DEFINITION} 
    BCS_BENCHMARKCBSESDF_PR=2.4f 
    BCS_SESDF_MAXNEIGHBORPERATOM=288 
    BCS_SSESDF_MAXNEIGHBORPERATOM=288
)
target_link_libraries(BCSBenchmarkCBSESDF2-4 PRIVATE sesdf288)
target_compile_options(BCSBenchmarkCBSESDF2-4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    --relocatable-device-code=true
    --extended-lambda
    -Xcudafe 
    --diag_suppress=esa_on_defaulted_function_ignored
    >
)

add_dependencies(BCSBenchmarkCBSESDF2-4 BCS_SAMPLES)
