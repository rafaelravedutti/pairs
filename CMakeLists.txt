cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pairs)
endif()

# Set default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, etc.)")
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -S -fverbose-asm")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

option(PAIRS_BUILD_WITH_WALBERLA    "Enable waLBerla support for using BlockForest domain partitioning and dynamic load balancing" OFF)
option(PAIRS_BUILD_WITH_CUDA        "PAIRS_BUILD_WITH_CUDA"     OFF)
option(PAIRS_BUILD_WITH_LIKWID      "PAIRS_BUILD_WITH_LIKWID"   OFF)
option(PAIRS_BUILD_EXAMPLES         "PAIRS_BUILD_EXAMPLES"      OFF)
option(PAIRS_BUILD_BENCHMARKS       "PAIRS_BUILD_BENCHMARKS"    OFF)

include(FetchContent)

#================================================================================
# RUNTIME_COMMON_FILES
#================================================================================
set(RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
set(RUNTIME_COMMON_FILES
    ${RUNTIME_DIR}/pairs.cpp
    ${RUNTIME_DIR}/utility/copper_fcc_lattice.cpp
    ${RUNTIME_DIR}/utility/create_body.cpp
    ${RUNTIME_DIR}/utility/dem_sc_grid.cpp
    ${RUNTIME_DIR}/utility/read_from_file.cpp
    ${RUNTIME_DIR}/utility/stats.cpp
    ${RUNTIME_DIR}/utility/thermo.cpp
    ${RUNTIME_DIR}/utility/timing.cpp
    ${RUNTIME_DIR}/utility/vtk.cpp
    ${RUNTIME_DIR}/domain/regular_6d_stencil.cpp
)

#================================================================================
# pairs runtime library (shared by all generated codes)
#================================================================================
add_library(pairsrt STATIC ${RUNTIME_COMMON_FILES})
target_include_directories(pairsrt PUBLIC ${RUNTIME_DIR})

set_target_properties(pairsrt PROPERTIES 
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 20
)

#================================================================================
# Install pairs python package (optionally using the specified python executable)
#================================================================================
if(NOT DEFINED PAIRS_PYTHON_EXECUTABLE)
    set(PAIRS_PYTHON_EXECUTABLE "" CACHE PATH "Python executable to install pairs")
endif()

if(PAIRS_PYTHON_EXECUTABLE STREQUAL "")
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PAIRS_PYTHON_EXECUTABLE ${Python_EXECUTABLE} CACHE PATH "Python executable to install pairs" FORCE)
endif()

message(STATUS "Installing pairs")

execute_process(
    COMMAND ${PAIRS_PYTHON_EXECUTABLE} -m pip install --force-reinstall --editable . -q
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE PAIRS_PIP_RESULT
)

if(NOT PAIRS_PIP_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install P4IRS Python package.")
endif()

message(STATUS "Installing pairs - done")
#================================================================================
# MPI
#================================================================================
find_package(MPI REQUIRED)
target_include_directories(pairsrt PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(pairsrt PUBLIC ${MPI_LIBRARIES})

#================================================================================
# CUDA
#================================================================================
if(PAIRS_BUILD_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 -DDEBUG")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()

    if(NOT DEFINED ENABLE_GPU_DIRECT)
        set(ENABLE_GPU_DIRECT ON "Enable GPU Direct (default: ON when PAIRS_BUILD_WITH_CUDA is ON)" FORCE)
    else()
         # User choice is respected here if they opt to PAIRS_BUILD_WITH_CUDA without GPU Direct
        set(ENABLE_GPU_DIRECT ${ENABLE_GPU_DIRECT} "Enable GPU Direct (user-defined)" FORCE)
    endif()

    if(ENABLE_GPU_DIRECT)
        target_compile_definitions(pairsrt PRIVATE ENABLE_CUDA_AWARE_MPI)
    endif()

    target_link_libraries(pairsrt PUBLIC CUDA::cudart)

    target_sources(pairsrt PRIVATE ${RUNTIME_DIR}/devices/cuda.cu)
    target_compile_definitions(pairsrt PUBLIC PAIRS_TARGET_CUDA)

    set_target_properties(pairsrt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

else()
    if(ENABLE_GPU_DIRECT)
        message(FATAL_ERROR "Invalid combination: ENABLE_GPU_DIRECT requires PAIRS_BUILD_WITH_CUDA to be ON")
    endif()

    target_sources(pairsrt PRIVATE ${RUNTIME_DIR}/devices/dummy.cpp)
endif()

#================================================================================
# WALBERLA
#================================================================================
if(PAIRS_BUILD_WITH_WALBERLA)
    target_compile_definitions(pairsrt PUBLIC USE_WALBERLA)

    set(RUNTIME_WALBERLA_FILES ${RUNTIME_DIR}/domain/block_forest.cpp)

    # TODO: Generate the host/device functions for computing weights
    if(COMPILE_CUDA)
        list(APPEND RUNTIME_WALBERLA_FILES ${RUNTIME_DIR}/domain/boundary_weights.cu)
    else()
        list(APPEND RUNTIME_WALBERLA_FILES ${RUNTIME_DIR}/domain/boundary_weights.cpp)
    endif()
        
    target_sources(pairsrt PRIVATE ${RUNTIME_WALBERLA_FILES})

    # Skip building a separate waLBerla if P4IRS is itself 
    # being built inside a waLBerla CMake environment
    if(NOT TARGET walberla::blockforest)
        message(STATUS "walberla::blockforest was not found. ")
        message(STATUS "Fetching waLBerla")
        FetchContent_Declare(
            walberla
            GIT_REPOSITORY https://i10git.cs.fau.de/walberla/walberla.git
            GIT_TAG 65fcc780c8843f982b23e3b1f14de6408982c5d8
        )
        FetchContent_MakeAvailable(walberla)
        message(STATUS "Fetching waLBerla - done")
    endif()

    target_link_libraries(pairsrt PUBLIC walberla::blockforest)
endif()


#================================================================================
# LIKWID
#================================================================================
if(PAIRS_BUILD_WITH_LIKWID)
    find_library( LIKWID_LIB likwid HINTS $ENV{LIKWID_LIBDIR} $ENV{LIKWID_ROOT}/lib )
    find_path( LIKWID_INCLUDE_DIR likwid.h HINTS $ENV{LIKWID_INCDIR} $ENV{LIKWID_ROOT}/include )

    if(LIKWID_LIB AND LIKWID_INCLUDE_DIR)
        target_compile_options(pairsrt PRIVATE -DLIKWID_PERFMON -pthread)

        target_include_directories(pairsrt PRIVATE ${LIKWID_INCLUDE_DIR})
        target_link_libraries(pairsrt PRIVATE ${LIKWID_LIB})
    else()
        message(FATAL_ERROR "LIKWID library was not found!")
    endif()

endif()

#================================================================================
# Load pairsFunctions
#================================================================================

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(pairsFunctions)


#================================================================================
# Apps
#================================================================================
add_subdirectory(apps)
