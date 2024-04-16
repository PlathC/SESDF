file(GLOB_RECURSE BCS_SAMPLES samples/*.*)
add_custom_target(BCS_SAMPLES COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../samples/ ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/samples/)
set_property(TARGET BCS_SAMPLES APPEND PROPERTY SOURCES ${BCS_SAMPLES}) 

add_subdirectory(benchmarks)
add_subdirectory(circles)
add_subdirectory(detailed)
add_subdirectory(memory)
