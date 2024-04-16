add_executable(BCSCirclesAnalysis "main.cu")
target_compile_features(BCSCirclesAnalysis PRIVATE cxx_std_17)
get_compute_architecture(bcs_circleanalysis_compute_arch)
set_target_properties(BCSCirclesAnalysis PROPERTIES CUDA_ARCHITECTURES ${bcs_circleanalysis_compute_arch})
set_target_properties(BCSCirclesAnalysis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSCirclesAnalysis PRIVATE ${BCS_COMPILATION_DEFINITION})
target_link_libraries(BCSCirclesAnalysis PRIVATE sesdf CUDA::cupti)
target_compile_options(BCSCirclesAnalysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    --relocatable-device-code=true
    --extended-lambda
    -Xcudafe 
    --diag_suppress=esa_on_defaulted_function_ignored
    >
)

add_dependencies(BCSCirclesAnalysis BCS_SAMPLES)
