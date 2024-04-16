# Load libraries

if (NOT TARGET SDL2::SDL2)
    option(SDL2_DISABLE_UNINSTALL "" ON) # Name clash between SDL2 and chemfiles
    add_subdirectory(SDL)
endif ()

if (NOT TARGET chemfiles)
    add_subdirectory(chemfiles)

    set(BCS_EXTERN_INCLUDES
            ${BCS_EXTERN_INCLUDES}
            "${CMAKE_CURRENT_SOURCE_DIR}/chemfiles/include"
    )
endif ()

# since chemfiles uses fmt this avoid to get the target multiples times
if (NOT TARGET fmt)
    add_subdirectory(fmt)
endif ()

if (NOT TARGET gl3w)
    add_subdirectory(gl3w)

    set(BCS_EXTERN_INCLUDES
            ${BCS_EXTERN_INCLUDES}
            "${CMAKE_CURRENT_SOURCE_DIR}/gl3w"
    )
endif ()

if (NOT TARGET cuda_helper)
    add_subdirectory(cuda_helper)

    set(BCS_EXTERN_INCLUDES
            ${BCS_EXTERN_INCLUDES}
            "${CMAKE_CURRENT_SOURCE_DIR}/cuda_helper"
    )
endif ()

if (NOT TARGET glm)
    add_subdirectory(glm)
endif ()

# Prepare data for parent scope
set(BCS_EXTERN_SOURCES ${BCS_EXTERN_SOURCES} ${GL3W_SOURCES})
set(BCS_EXTERN_INCLUDES
        ${BCS_EXTERN_INCLUDES}
        "${CMAKE_CURRENT_SOURCE_DIR}/magic_enum/include"
)

set(BCS_EXTERN_LIBRARIES
        SDL2::SDL2
        SDL2::SDL2main
        glm::glm
        fmt
        chemfiles
        cuda_helper
        gl3w
)

if (SESDF_WITH_ANALYSIS AND NOT TARGET ContourBuildup)
    add_subdirectory(ContourBuildup)

    set(BCS_EXTERN_LIBRARIES
            ${BCS_EXTERN_LIBRARIES}

            ContourBuildup
    )
endif ()


set(BCS_EXTERN_INCLUDES ${BCS_EXTERN_INCLUDES} PARENT_SCOPE)
set(BCS_EXTERN_SOURCES ${BCS_EXTERN_SOURCES} PARENT_SCOPE)
set(BCS_EXTERN_LIBRARIES ${BCS_EXTERN_LIBRARIES} PARENT_SCOPE)
