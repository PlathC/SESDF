project(gl3w C)

file(GLOB_RECURSE gl3w_sources *.h *.c)

add_library(gl3w STATIC ${gl3w_sources})
target_include_directories(gl3w PUBLIC .)
