project(ContourBuildup CUDA)

add_library(ContourBuildup 
    STATIC 

    src/particleSystem.cu
)

target_include_directories(ContourBuildup PUBLIC include/)

if(NOT TARGET gl3w)
    message( FATAL_ERROR "Can't find gl3w target" )
endif()

target_link_libraries(ContourBuildup PRIVATE gl3w)

if(NOT TARGET cuda_helper)
    message( FATAL_ERROR "Can't find cuda_helper target" )
endif()

target_link_libraries(ContourBuildup PRIVATE cuda_helper)

get_compute_architecture(cb_cuda_architecture)
set_target_properties(ContourBuildup PROPERTIES CUDA_ARCHITECTURES ${cb_cuda_architecture})
