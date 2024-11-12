cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)
# cmake_policy(SET CMP0074 NEW)
project(pairs CXX)

set(TESTCASE ${TESTCASE} CACHE STRING "Select the testcase from the following: md, dem")

if(NOT TESTCASE)
    set(TESTCASE md CACHE STRING "Select the testcase from the following: md, dem" FORCE)
endif()

set(TARGET_BIN ${TESTCASE})

# Set default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, etc.)" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

string(TOLOWER "${TESTCASE}" TESTCASE)
message(STATUS "Selected testcase: ${TESTCASE}")

option(USE_WALBERLA "USE_WALBERLA" ON)
option(USE_MPI "USE_MPI" ON)
option(COMPILE_CUDA "COMPILE_CUDA" ON)
option(ENABLE_GPU_DIRECT "ENABLE_GPU_DIRECT" ON)
option(GENERATE_WHOLE_PROGRAM "GENERATE_WHOLE_PROGRAM" OFF)

if(USE_WALBERLA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_WALBERLA_LOAD_BALANCING -DUSE_WALBERLA")
    find_package(waLBerla REQUIRED)
    add_subdirectory(${walberla_SOURCE_DIR} ${walberla_BINARY_DIR} EXCLUDE_FROM_ALL)
    waLBerla_import()
endif()

set(GEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/runtime/interfaces/last_generated.hpp)
set(GEN_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/runtime/interfaces)

set(RUNTIME_COMMON_FILES
    runtime/pairs.cpp
    runtime/copper_fcc_lattice.cpp
    runtime/create_body.cpp
    runtime/dem_sc_grid.cpp
    runtime/read_from_file.cpp
    runtime/stats.cpp
    runtime/thermo.cpp
    runtime/timing.cpp
    runtime/vtk.cpp
    runtime/domain/regular_6d_stencil.cpp)
    
set(RUNTIME_WALBERLA_FILES
    # runtime/boundary_weights.cpp  # avoid compiling this for now. TODO: generate the host and device functions
    runtime/domain/block_forest.cpp)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE STRING "Python executable not set.")

if(NOT GENERATE_WHOLE_PROGRAM)
    file(
        COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp
        DESTINATION ${CMAKE_BINARY_DIR})

    set(EXEC_FILES ${CMAKE_BINARY_DIR}/main.cpp ${RUNTIME_COMMON_FILES})
else()
    set(EXEC_FILES ${RUNTIME_COMMON_FILES})
endif()

if(USE_WALBERLA)
    set(EXEC_FILES ${EXEC_FILES} ${RUNTIME_WALBERLA_FILES})
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
endif()

file(COPY runtime DESTINATION ${CMAKE_BINARY_DIR})
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
if(EXISTS ${OUTPUT_DIR})
    file(REMOVE_RECURSE ${OUTPUT_DIR})
endif()

file(MAKE_DIRECTORY ${OUTPUT_DIR})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} setup.py build
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} setup.py install --user
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA compilation
if(COMPILE_CUDA)
    cmake_policy(SET CMP0074 NEW)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(GEN_SOURCES "${CMAKE_BINARY_DIR}/${TESTCASE}.cu")
    set(CUDA_ARCH ${CUDA_ARCH} CACHE STRING "CUDA_ARCH environment variable must be set.")
    set(TARGET_ARG "gpu")

    # Default arch is 80
    if(NOT CUDA_ARCH)
        set(CUDA_ARCH 80)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 -DDEBUG")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()

    if(USE_WALBERLA)
        waLBerla_add_executable(
            NAME ${TARGET_BIN}
            FILES ${EXEC_FILES} ${GEN_SOURCES}
            DEPENDS blockforest core pe)
    else()
        add_executable(${TARGET_BIN} ${GEN_SOURCES} ${EXEC_FILES})
    endif()

    if(ENABLE_GPU_DIRECT)
        target_compile_definitions(${TARGET_BIN} PRIVATE ENABLE_CUDA_AWARE_MPI)
    endif()

    add_library(runtime STATIC runtime/devices/cuda.cu)
    target_compile_features(runtime PUBLIC cxx_std_11)
    set_target_properties(runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_compile_definitions(runtime PRIVATE PAIRS_TARGET_CUDA)
    target_include_directories(runtime PRIVATE ${CUDA_INCLUDE_DIRS})

    set_target_properties(${TARGET_BIN} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_compile_definitions(${TARGET_BIN} PRIVATE PAIRS_TARGET_CUDA)
    target_include_directories(${TARGET_BIN} PRIVATE ${CUDA_INCLUDE_DIRS} ${GEN_HEADER_DIR})

# CPU compilation
else()
    set(GEN_SOURCES "${CMAKE_BINARY_DIR}/${TESTCASE}.cpp")
    set(TARGET_ARG "cpu")

    if(USE_WALBERLA)
        waLBerla_add_executable(
            NAME ${TARGET_BIN}
            FILES ${EXEC_FILES} ${GEN_SOURCES}
            DEPENDS blockforest core pe)
    else()
        add_executable(${TARGET_BIN} ${GEN_SOURCES} ${EXEC_FILES})
    endif()

    add_library(runtime STATIC runtime/devices/dummy.cpp)
    target_include_directories(${TARGET_BIN} PRIVATE ${GEN_HEADER_DIR})
endif()

target_link_libraries(${TARGET_BIN} runtime)

add_custom_command(
    OUTPUT ${GEN_SOURCES} ${GEN_HEADER}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py ${TARGET_ARG}
    COMMENT "Generating code with P4IRS"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py)

add_custom_target(generated_code DEPENDS ${GEN_SOURCES} ${GEN_HEADER})
add_dependencies(${TARGET_BIN} generated_code)

target_link_libraries(${TARGET_BIN} ${CMAKE_EXE_LINKER_FLAGS})
set_target_properties(${TARGET_BIN} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TARGET_BIN} PROPERTIES CXX_STANDARD 17)

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(${TARGET_BIN} ${MPI_LIBRARIES})
endif()

if(LIKWID_DIR)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIKWID_PERFMON -pthread -L${LIKWID_DIR}/lib -llikwid ")
    include_directories(${LIKWID_DIR}/include)
endif()
