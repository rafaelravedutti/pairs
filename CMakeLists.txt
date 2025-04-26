cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pairs CXX)
# Set default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, etc.)" FORCE)
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -S -fverbose-asm")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

option(USE_MPI                      "USE_MPI" ON)
option(USE_WALBERLA                 "Enable waLBerla support for using BlockForest domain partitioning and dynamic load balancing" OFF)
option(COMPILE_CUDA                 "COMPILE_CUDA" OFF)
set(USER_SOURCE_FILES "" CACHE STRING "List of source files to compile (semicolon-separated)")

set(INPUT_SCRIPT ${INPUT_SCRIPT} CACHE PATH "The input python script triggering code generation")
if(NOT EXISTS ${INPUT_SCRIPT})
    message(FATAL_ERROR "INPUT_SCRIPT doesn't exist! Specify it with -DINPUT_SCRIPT=/path/to/script.py")
endif()
get_filename_component(INPUT_SCRIPT_NAME ${INPUT_SCRIPT} NAME_WE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

#================================================================================
# Setup directories =============================================================
#================================================================================
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output)
if(EXISTS ${OUTPUT_DIR})
    file(REMOVE_RECURSE ${OUTPUT_DIR})
endif()
file(MAKE_DIRECTORY ${OUTPUT_DIR})

#================================================================================
# Generated header (internally used by runtime files) ===========================
#================================================================================
# TODO: Unify all interfaces
set(GEN_INTERFACE_DIR ${CMAKE_CURRENT_BINARY_DIR}/internal_interfaces)
set(GEN_INTERFACE_HEADER ${GEN_INTERFACE_DIR}/last_generated.hpp)
file(MAKE_DIRECTORY ${GEN_INTERFACE_DIR})

#================================================================================
# RUNTIME_COMMON_FILES ==========================================================
#================================================================================
set(RUNTIME_COMMON_FILES
    runtime/pairs.cpp
    runtime/utility/copper_fcc_lattice.cpp
    runtime/utility/create_body.cpp
    runtime/utility/dem_sc_grid.cpp
    runtime/utility/read_from_file.cpp
    runtime/utility/stats.cpp
    runtime/utility/thermo.cpp
    runtime/utility/timing.cpp
    runtime/utility/vtk.cpp
    runtime/domain/regular_6d_stencil.cpp)

#================================================================================
# PAIRS_TARGET ==================================================================
#================================================================================
set(PAIRS_TARGET "pairs")

# PAIRS dependencies 
set(PAIRS_LINK_LIBRARIES)
set(PAIRS_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR})

# The target can either be an executable or a static library
if(USER_SOURCE_FILES)
    add_executable(${PAIRS_TARGET} ${RUNTIME_COMMON_FILES})
else()
    add_library(${PAIRS_TARGET} STATIC ${RUNTIME_COMMON_FILES})

    # TODO: Get the location with $<TARGET_FILE> generator expression (at build time)
    list(APPEND PAIRS_LINK_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/lib${PAIRS_TARGET}.a")
endif()

# Include P4IRS 'runtime' dir
target_include_directories(${PAIRS_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
list(APPEND PAIRS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

set_target_properties(${PAIRS_TARGET} PROPERTIES 
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
)

#================================================================================
# USER_SOURCE_FILES =============================================================
#================================================================================
if(USER_SOURCE_FILES)
    foreach(file ${USER_SOURCE_FILES})
        if(NOT EXISTS ${file})
            message(FATAL_ERROR "File '${file}' does not exist!")
        endif()
    endforeach()
    target_sources(${PAIRS_TARGET} PRIVATE ${USER_SOURCE_FILES})
endif()

#================================================================================
# waLBerla ======================================================================
#================================================================================
if(USE_WALBERLA)
    target_compile_definitions(${PAIRS_TARGET} PUBLIC USE_WALBERLA)

    set(RUNTIME_WALBERLA_FILES
        runtime/domain/block_forest.cpp
    )

    # TODO: Generate the host/device functions for computing weights
    if(COMPILE_CUDA)
        list(APPEND RUNTIME_WALBERLA_FILES runtime/domain/boundary_weights.cu)
    else()
        list(APPEND RUNTIME_WALBERLA_FILES runtime/domain/boundary_weights.cpp)
    endif()
    
    target_sources(${PAIRS_TARGET} PRIVATE ${RUNTIME_WALBERLA_FILES})

    add_subdirectory(external/walberla EXCLUDE_FROM_ALL)
    waLBerla_import()

    # Recent issue from the waLBerla side: warnings about empty CUDA_ARCHITECTURES for walberla_gpu 
    target_link_libraries(${PAIRS_TARGET} PUBLIC 
        walberla::blockforest 
        walberla::domain_decomposition 
        walberla::core 
        stdc++fs
    )

    # TODO: Get the location with $<TARGET_FILE> generator expression (at build time)
    list(APPEND PAIRS_LINK_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/external/walberla/src/blockforest/libwalberla_blockforest.a)
    list(APPEND PAIRS_LINK_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/external/walberla/src/domain_decomposition/libwalberla_domain_decomposition.a)
    list(APPEND PAIRS_LINK_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/external/walberla/src/core/libwalberla_core.a)
    list(APPEND PAIRS_LINK_LIBRARIES stdc++fs)
endif()

#================================================================================
# Install pairs python package ==================================================
#================================================================================
set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE STRING "Python executable")

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
endif()

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} setup.py build
    OUTPUT_QUIET
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} setup.py install --user
    OUTPUT_QUIET
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#================================================================================
# CUDA ==========================================================================
#================================================================================
if(COMPILE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(GEN_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${INPUT_SCRIPT_NAME}.cu")
    set(CUDA_ARCH ${CUDA_ARCH} CACHE STRING "CUDA_ARCH environment variable must be set.")
    set(TARGET_ARG "gpu")

    # Default arch is 80
    if(NOT CUDA_ARCH)
        set(CUDA_ARCH 80)
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 -DDEBUG")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()

    if(NOT DEFINED ENABLE_GPU_DIRECT)
        set(ENABLE_GPU_DIRECT ON "Enable GPU Direct (default: ON when COMPILE_CUDA is ON)" FORCE)
    else()
         # User choice is respected here if they opt to COMPILE_CUDA without GPU Direct
        set(ENABLE_GPU_DIRECT ${ENABLE_GPU_DIRECT} "Enable GPU Direct (user-defined)" FORCE)
    endif()

    if(ENABLE_GPU_DIRECT)
        target_compile_definitions(${PAIRS_TARGET} PRIVATE ENABLE_CUDA_AWARE_MPI)
    endif()

    target_sources(${PAIRS_TARGET} PRIVATE runtime/devices/cuda.cu)
    target_compile_definitions(${PAIRS_TARGET} PUBLIC PAIRS_TARGET_CUDA)
    target_include_directories(${PAIRS_TARGET} PUBLIC ${CUDA_INCLUDE_DIRS})
    list(APPEND PAIRS_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

    set_target_properties(${PAIRS_TARGET} PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH})

    target_link_libraries(${PAIRS_TARGET} PUBLIC ${CUDA_LIBRARIES})
    list(APPEND PAIRS_LINK_LIBRARIES ${CUDA_LIBRARIES})
else()
    if(ENABLE_GPU_DIRECT)
        message(FATAL_ERROR "Invalid combination: ENABLE_GPU_DIRECT requires COMPILE_CUDA to be ON")
    endif()

    set(GEN_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${INPUT_SCRIPT_NAME}.cpp")
    set(TARGET_ARG "cpu")
    target_sources(${PAIRS_TARGET} PRIVATE runtime/devices/dummy.cpp)
endif()

#================================================================================
# Generate code and add them to PAIRS_TARGET ====================================
#================================================================================
add_custom_command(
    OUTPUT ${GEN_SOURCES} ${GEN_INTERFACE_HEADER}
    COMMAND ${PYTHON_EXECUTABLE} ${INPUT_SCRIPT} ${TARGET_ARG}
    COMMENT "Generating code with P4IRS"
    DEPENDS ${INPUT_SCRIPT})

add_custom_target(generated_code DEPENDS ${GEN_SOURCES} ${GEN_INTERFACE_HEADER})
add_dependencies(${PAIRS_TARGET} generated_code)

target_sources(${PAIRS_TARGET} PRIVATE ${GEN_SOURCES})
target_include_directories(${PAIRS_TARGET} PRIVATE 
    ${GEN_INTERFACE_DIR}            # Interface header USED INTERNALLY by pairs is located here
    ${CMAKE_CURRENT_BINARY_DIR}     # Generated source and header FOR USER is located here
)    

#================================================================================
# MPI ===========================================================================
#================================================================================
if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(${PAIRS_TARGET} PUBLIC ${MPI_LIBRARIES})
    list(APPEND PAIRS_LINK_LIBRARIES "${MPI_LIBRARIES}")
    list(APPEND PAIRS_INCLUDE_DIRS "${MPI_INCLUDE_PATH}")
endif()

#================================================================================
# LIKWID ========================================================================
#================================================================================
if(LIKWID_DIR)
    # Example: -DLIKWID_DIR=/apps/likwid/5.4.0

    target_compile_options(${PAIRS_TARGET} PRIVATE -DLIKWID_PERFMON -pthread)

    target_link_libraries(${PAIRS_TARGET} PRIVATE ${LIKWID_DIR}/lib/liblikwid.so)
    list(APPEND PAIRS_LINK_LIBRARIES ${LIKWID_DIR}/lib/liblikwid.so)

    include_directories(${LIKWID_DIR}/include)
    list(APPEND PAIRS_INCLUDE_DIRS "${LIKWID_DIR}/include")
endif()

#================================================================================
# config file ===================================================================
#================================================================================
configure_file ( "${CMAKE_SOURCE_DIR}/cmake/pairs-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pairs-config.cmake")
