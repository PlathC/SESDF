add_executable(BCSDetailedSESDF "sesdf_main.cu" "grid_benchmark.cu" "sesdf_benchmark.cu")
target_compile_features(BCSDetailedSESDF PRIVATE cxx_std_17)
get_compute_architecture(bcs_detailed_compute_arch)
set_target_properties(BCSDetailedSESDF PROPERTIES CUDA_ARCHITECTURES ${bcs_detailed_compute_arch})
set_target_properties(BCSDetailedSESDF PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSDetailedSESDF PRIVATE ${BCS_COMPILATION_DEFINITION})
target_link_libraries(BCSDetailedSESDF PRIVATE sesdf)
target_compile_options(BCSDetailedSESDF PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    --relocatable-device-code=true
    --extended-lambda
    -Xcudafe 
    --diag_suppress=esa_on_defaulted_function_ignored
    >
)

add_dependencies(BCSDetailedSESDF BCS_SAMPLES)

add_executable(BCSDetailedCB "cb_main.cu" "cb_benchmark.cu")
target_compile_features(BCSDetailedCB PRIVATE cxx_std_17)
get_compute_architecture(bcs_detailed_compute_arch)
set_target_properties(BCSDetailedCB PROPERTIES CUDA_ARCHITECTURES ${bcs_detailed_compute_arch})
set_target_properties(BCSDetailedCB PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSDetailedCB PRIVATE ${BCS_COMPILATION_DEFINITION})
target_link_libraries(BCSDetailedCB PRIVATE sesdf)
target_compile_options(BCSDetailedCB PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    --relocatable-device-code=true
    --extended-lambda
    -Xcudafe 
    --diag_suppress=esa_on_defaulted_function_ignored
    >
)

add_dependencies(BCSDetailedCB BCS_SAMPLES)
