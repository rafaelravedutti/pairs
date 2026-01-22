#================================================================================
# This function generates sources from the Python script and compiles 
# them as a static library. 
#
# Arguments:
#   GEN_LIB         Name of the generated library. This is a CMake target, thus 
#                   the name must be unique across the whole project.
#   SCRIPT          Path to the Python script that triggers code generation.
#   OUTPUT_DIR      Directory where the generated sources will be written.
#================================================================================

function(pairs_generate_lib GEN_LIB SCRIPT OUTPUT_DIR)
    # Make SCRIPT path absolute if needed (from the source dir)
    if(NOT IS_ABSOLUTE "${SCRIPT}")
        set(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}")
    endif()

    if(NOT EXISTS "${SCRIPT}")
        message(FATAL_ERROR "P4IRS input script not found: '${SCRIPT}'")
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
        set(GEN_SOURCES "${GEN_USER_INTERFACE_DIR}/${GEN_LIB}.cu")
        set(TARGET_ARG "gpu")
    else()
        set(GEN_SOURCES "${GEN_USER_INTERFACE_DIR}/${GEN_LIB}.cpp")
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

    set(CGEN_TARGET "pairs_cgen_${GEN_LIB}" )

    # Generate code
    add_custom_command(
        OUTPUT ${GEN_SOURCES} ${GEN_INTERNAL_INTERFACE_HEADER}
        COMMAND ${PYTHON_EXECUTABLE} ${SCRIPT} 
                --interface-name ${GEN_LIB}
                --target ${TARGET_ARG} 
                --output-dir ${OUTPUT_DIR}
                --debug ${DEBUG_ARG}
        COMMENT "P4IRS: Generating code for the library '${GEN_LIB}' using the script '${SCRIPT}'."
        DEPENDS ${SCRIPT}
        BYPRODUCTS ${GEN_SOURCES} ${GEN_INTERNAL_INTERFACE_HEADER}
    )
        
    add_custom_target(${CGEN_TARGET} DEPENDS ${GEN_SOURCES} ${GEN_INTERNAL_INTERFACE_HEADER})

    # The generated code is itself built as a separate library
    add_library(${GEN_LIB} STATIC ${GEN_SOURCES})

    # Add depenency on the generated code (triggers regeneration on script updates)
    add_dependencies(${GEN_LIB} ${CGEN_TARGET})

    # Link the generated lib to pairs runime lib (the runtime lib is shared by all generated codes)
    target_link_libraries(${GEN_LIB} PUBLIC pairsrt)

    # Include the generated user-facing header for the generated lib
    target_include_directories(${GEN_LIB} PUBLIC ${GEN_USER_INTERFACE_DIR})

    # Include the generated internal header in the pairs runtime lib (TODO: to be removed)
    target_include_directories(pairsrt PRIVATE  ${GEN_INTERNAL_INTERFACE_DIR})
endfunction()


#================================================================================
# This function links a given CMake target to a given generated pairs library. 
#
# Arguments:
#   TARGET          The CMake target
#   GEN_LIB         The generated library that was built using pairs_generate_lib
#================================================================================

function(pairs_attach_to_target TARGET GEN_LIB) 
    if(NOT TARGET ${GEN_LIB})
        message(FATAL_ERROR "The CMake target '${GEN_LIB}' does not exist.\n"
                "Make sure you call pairs_generate_lib() to generate the library first.")
    endif()

    target_link_libraries(${TARGET} PUBLIC ${GEN_LIB})
endfunction()