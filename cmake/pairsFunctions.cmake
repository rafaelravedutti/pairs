#================================================================================
# This function generates sources from the Python script and compiles 
# them as a static library. 
#
# Arguments:
#   GEN_LIB     [required]  Name of the generated library. This is a CMake target, thus 
#                           the name must be unique across the whole project.
#   SCRIPT      [required]  Path to the Python script that triggers code generation.
#   OUTPUT_DIR  [optional]  Directory where the generated sources will be written.
#================================================================================

function(pairs_generate_lib)
    set(oneValueArgs GEN_LIB SCRIPT OUTPUT_DIR)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})

    if(NOT ARG_GEN_LIB OR NOT ARG_SCRIPT)
        message(FATAL_ERROR "pairs_generate_lib requires GEN_LIB and SCRIPT")
    endif()

    # Make SCRIPT path absolute if needed (from the source dir)
    if(NOT IS_ABSOLUTE "${ARG_SCRIPT}")
        set(ARG_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SCRIPT}")
    endif()

    if(NOT EXISTS "${ARG_SCRIPT}")
        message(FATAL_ERROR "P4IRS input script not found: '${ARG_SCRIPT}'")
    endif()

    # Default output directory
    if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
        set(OUTPUT_DIR "gen_${ARG_GEN_LIB}")
    endif()

    # Make OUTPUT_DIR absolute if needed (from the build dir)
    if(NOT IS_ABSOLUTE "${OUTPUT_DIR}")
        set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_DIR}")
    endif()

    # Create OUTPUT_DIR if it doesn't exist
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Output 1: User interface
    set(GEN_USER_INTERFACE_DIR ${OUTPUT_DIR})
    file(MAKE_DIRECTORY ${GEN_USER_INTERFACE_DIR})

    if(PAIRS_BUILD_WITH_CUDA)
        set(GEN_SOURCES "${GEN_USER_INTERFACE_DIR}/${ARG_GEN_LIB}.cu")
        set(TARGET_ARG "gpu")
    else()
        set(GEN_SOURCES "${GEN_USER_INTERFACE_DIR}/${ARG_GEN_LIB}.cpp")
        set(TARGET_ARG "cpu")
    endif()

    # Output 2: Internal interface (TODO: to be removed)
    set(GEN_INTERNAL_INTERFACE_DIR "${OUTPUT_DIR}/internal")
    set(GEN_INTERNAL_INTERFACE_HEADER ${GEN_INTERNAL_INTERFACE_DIR}/last_generated.hpp)
    file(MAKE_DIRECTORY ${GEN_INTERNAL_INTERFACE_DIR})

    # Debug arg for code generation
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(DEBUG_ARG 1)
    else()
        set(DEBUG_ARG 0)
    endif()

    set(CGEN_TARGET "pairs_cgen_${ARG_GEN_LIB}" )

    # Generate code
    add_custom_command(
        OUTPUT ${GEN_SOURCES} ${GEN_INTERNAL_INTERFACE_HEADER}
        COMMAND ${PAIRS_PYTHON_EXECUTABLE} ${ARG_SCRIPT} 
                --interface-name ${ARG_GEN_LIB}
                --target ${TARGET_ARG} 
                --output-dir ${OUTPUT_DIR}
                --debug ${DEBUG_ARG}
        COMMENT "P4IRS: Generating code for the library '${ARG_GEN_LIB}' using the script '${ARG_SCRIPT}'."
        DEPENDS ${ARG_SCRIPT}
    )
        
    add_custom_target(${CGEN_TARGET} DEPENDS ${GEN_SOURCES} ${GEN_INTERNAL_INTERFACE_HEADER})

    # The generated code is itself built as a separate library
    add_library(${ARG_GEN_LIB} STATIC ${GEN_SOURCES})

    if(PAIRS_BUILD_WITH_CUDA)
        # Separable compilation is required here since the cuda kernels in the generated code
        # call device functions that are defined in a separate .cu file in the pairsrt library
        set_target_properties(${ARG_GEN_LIB} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    # Add depenency on the generated code (triggers regeneration on script updates)
    add_dependencies(${ARG_GEN_LIB} ${CGEN_TARGET})

    # Link the generated lib to pairs runime lib (the runtime lib is shared by all generated codes)
    target_link_libraries(${ARG_GEN_LIB} PUBLIC pairsrt)

    # Include the generated user-facing header for the generated lib
    target_include_directories(${ARG_GEN_LIB} PUBLIC ${GEN_USER_INTERFACE_DIR})

    # Include the generated internal header in the pairs runtime lib (TODO: to be removed)
    target_include_directories(pairsrt PRIVATE  ${GEN_INTERNAL_INTERFACE_DIR})
endfunction()
