add_executable(BCSMemoryConsumption "main.cu")
target_compile_features(BCSMemoryConsumption PRIVATE cxx_std_17)
get_compute_architecture(bcs_memoryconsumption_compute_arch)
set_target_properties(BCSMemoryConsumption PROPERTIES CUDA_ARCHITECTURES ${bcs_memoryconsumption_compute_arch})
set_target_properties(BCSMemoryConsumption PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


target_compile_definitions(BCSMemoryConsumption PRIVATE ${BCS_COMPILATION_DEFINITION}
        BCS_MEMORYCONSUMPTION_PR=1.4f
        BCS_SESDF_MAXNEIGHBORPERATOM=128
        BCS_SSESDF_MAXNEIGHBORPERATOM=128
)
target_link_libraries(BCSMemoryConsumption PRIVATE sesdf CUDA::cupti)
target_compile_options(BCSMemoryConsumption PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --extended-lambda
        -Xcudafe
        --diag_suppress=esa_on_defaulted_function_ignored
        >
)

add_dependencies(BCSMemoryConsumption BCS_SAMPLES)

add_executable(BCSMemoryConsumption2-4 "main.cu")
target_compile_features(BCSMemoryConsumption2-4 PRIVATE cxx_std_17)
get_compute_architecture(bcs_memoryconsumption24_compute_arch)
set_target_properties(BCSMemoryConsumption2-4 PROPERTIES CUDA_ARCHITECTURES ${bcs_memoryconsumption24_compute_arch})
set_target_properties(BCSMemoryConsumption2-4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(BCSMemoryConsumption2-4 PRIVATE ${BCS_COMPILATION_DEFINITION}
        BCS_MEMORYCONSUMPTION_PR=2.4f
        BCS_SESDF_MAXNEIGHBORPERATOM=288
        BCS_SSESDF_MAXNEIGHBORPERATOM=288
)
target_link_libraries(BCSMemoryConsumption2-4 PRIVATE sesdf288 CUDA::cupti)
target_compile_options(BCSMemoryConsumption2-4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --extended-lambda
        -Xcudafe
        --diag_suppress=esa_on_defaulted_function_ignored
        >
)

add_dependencies(BCSMemoryConsumption2-4 BCS_SAMPLES)
