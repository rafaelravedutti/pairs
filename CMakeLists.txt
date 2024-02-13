cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

project(pairs CXX)
#enable_testing()

set(TESTCASE ${TESTCASE} CACHE STRING "Select the testcase from the following: md, dem")

if(NOT TESTCASE)
    set(TESTCASE md CACHE STRING "Select the testcase from the following: md, dem" FORCE)
endif()

string(TOLOWER "${TESTCASE}" TESTCASE)
message(STATUS "Selected testcase: ${TESTCASE}")

option(USE_WALBERLA "USE_WALBERLA" OFF)
option(COMPILE_CUDA "COMPILE_CUDA" OFF)

set(CPU_SRC "${TESTCASE}.cpp")
set(GPU_SRC "${TESTCASE}.cu")
set(CPU_BIN "${TESTCASE}_cpu")
set(GPU_BIN "${TESTCASE}_gpu")

set(RUNTIME_COMMON_FILES
    runtime/pairs.cpp
    runtime/domain/block_forest.cpp
    runtime/domain/regular_6d_stencil.cpp)
 
if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(pairs ${MPI_LIBRARIES})
endif()

if(LIKWID_DIR)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIKWID_PERFMON -llikwid -pthread ")
    target_link_libraries(pairs -I${LIKWID_DIR}/include -L${LIKWID_DIR}/lib -llikwid)
    include_directories(${LIKWID_DIR}/include)
endif()

if(USE_WALBERLA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_WALBERLA_LOAD_BALANCING ")
    find_package(waLBerla REQUIRED)
    add_subdirectory(${walberla_SOURCE_DIR} ${walberla_BINARY_DIR} EXCLUDE_FROM_ALL)
    waLBerla_import()

    waLBerla_add_executable(
        NAME ${CPU_BIN}
        FILES ${CPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_CPU_FILES}
        DEPENDS blockforest core pe)
else()
    add_executable(${CPU_BIN} ${CPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_CPU_FILES})
endif()

add_library(runtime_cpu STATIC runtime/devices/dummy.cpp)
target_link_libraries(${CPU_BIN} PRIVATE runtime_cpu)

add_custom_command(OUTPUT ${CPU_SRC} COMMAND python examples/${TESTCASE}.py cpu)
target_link_libraries(${CPU_BIN} ${CMAKE_EXE_LINKER_FLAGS})
set_target_properties(${CPU_BIN} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${CPU_BIN} PROPERTIES CXX_STANDARD 14)

if(COMPILE_CUDA)
    find_package(CUDA REQUIRED)

    add_library(runtime_gpu STATIC runtime/devices/cuda.cu})
    target_compile_features(runtime_gpu PUBLIC cxx_std_11)
    set_target_properties(runtime_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    #set_property(TARGET ${GPU_BIN} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${GPU_BIN} PRIVATE particles)

    if(USE_WALBERLA)
        waLBerla_add_executable(
            NAME ${GPU_BIN}
            FILES ${GPU_SRC} ${RUNTIME_COMMON_FILES}
            DEPENDS blockforest core pe)
    else()
        add_executable(${GPU_BIN} ${GPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_GPU_FILES})
    endif()

    add_custom_command(OUTPUT ${GPU_SRC} COMMAND python examples/${TESTCASE}.py gpu)
endif()
