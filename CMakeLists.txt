cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

project(pairs CXX)
#enable_testing()

find_package(PythonInterp)
find_package(Python)

set(TESTCASE ${TESTCASE} CACHE STRING "Select the testcase from the following: md, dem")

if(NOT TESTCASE)
    set(TESTCASE md CACHE STRING "Select the testcase from the following: md, dem" FORCE)
endif()

set(CUDA_ARCH ${CUDA_ARCH} CACHE STRING "CUDA_ARCH environment variable must be set.")

if(NOT CUDA_ARCH)
    set(CUDA_ARCH sm_80)
endif()

string(TOLOWER "${TESTCASE}" TESTCASE)
message(STATUS "Selected testcase: ${TESTCASE}")

option(USE_WALBERLA "USE_WALBERLA" ON)
option(USE_MPI "USE_MPI" ON)
option(COMPILE_CUDA "COMPILE_CUDA" ON)
option(ENABLE_GPU_DIRECT "ENABLE_GPU_DIRECT" ON)

set(CPU_SRC "${TESTCASE}.cpp")
set(GPU_SRC "${TESTCASE}.cu")
set(CPU_BIN "${TESTCASE}_cpu")
set(GPU_BIN "${TESTCASE}_gpu")

set(RUNTIME_COMMON_FILES
    runtime/pairs.cpp
    runtime/domain/block_forest.cpp
    runtime/domain/regular_6d_stencil.cpp)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} setup.py build && ${PYTHON_EXECUTABLE} setup.py install --user
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE CMD_RESULT)

if(USE_WALBERLA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_WALBERLA_LOAD_BALANCING ")
    find_package(waLBerla REQUIRED)
    add_subdirectory(${walberla_SOURCE_DIR} ${walberla_BINARY_DIR} EXCLUDE_FROM_ALL)
    waLBerla_import()

    waLBerla_add_executable(
        NAME ${CPU_BIN}
        FILES ${CPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_CPU_FILES}
        DEPENDS blockforest core pe ${CPU_SRC} runtime_cpu)
else()
    add_executable(${CPU_BIN} ${CPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_CPU_FILES})
endif()

add_library(runtime_cpu STATIC runtime/devices/dummy.cpp)
#target_link_libraries(${CPU_BIN} runtime_cpu)

add_custom_command(
    OUTPUT ${CPU_SRC}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py cpu
    COMMENT "Generate CPU code"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py)

add_custom_target(gen_cpu DEPENDS ${CPU_SRC})
add_dependencies(${CPU_BIN} gen_cpu)

if(COMPILE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)

    if(USE_WALBERLA)
        waLBerla_add_executable(
            NAME ${GPU_BIN}
            FILES ${GPU_SRC} ${RUNTIME_COMMON_FILES}
            DEPENDS blockforest core pe ${GPU_SRC} runtime_gpu)
    else()
        add_executable(${GPU_BIN} ${GPU_SRC} ${RUNTIME_COMMON_FILES} ${RUNTIME_GPU_FILES})
    endif()

    if(ENABLE_GPU_DIRECT)
        target_compile_options(${GPU_BIN} PRIVATE -DENABLE_CUDA_AWARE_MPI)
    endif()

    add_library(runtime_gpu STATIC runtime/devices/cuda.cu)
    target_compile_features(runtime_gpu PUBLIC cxx_std_11)
    set_target_properties(runtime_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(${GPU_BIN} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)
    target_include_directories(${GPU_BIN} PRIVATE ${CUDA_INCLUDE_DIRS})
    set_target_properties(${GPU_BIN} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

    target_compile_options(runtime_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)
    target_include_directories(runtime_gpu PRIVATE ${CUDA_INCLUDE_DIRS})
    set_target_properties(runtime_gpu PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

    #set_property(TARGET ${GPU_BIN} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    #target_link_libraries(${GPU_BIN} runtime_gpu)

    add_custom_command(
        OUTPUT ${GPU_SRC}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py gpu
        COMMENT "Generate GPU code"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${TESTCASE}.py)

    add_custom_target(gen_gpu DEPENDS ${GPU_SRC})
    add_dependencies(${GPU_BIN} gen_gpu)
endif()

target_link_libraries(${CPU_BIN} ${CMAKE_EXE_LINKER_FLAGS})
set_target_properties(${CPU_BIN} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${CPU_BIN} PROPERTIES CXX_STANDARD 14)

target_link_libraries(${GPU_BIN} ${CMAKE_EXE_LINKER_FLAGS})
set_target_properties(${GPU_BIN} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${GPU_BIN} PROPERTIES CXX_STANDARD 14)

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(${CPU_BIN} ${MPI_LIBRARIES})
    target_link_libraries(${GPU_BIN} ${MPI_LIBRARIES})
endif()

if(LIKWID_DIR)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIKWID_PERFMON -pthread -L${LIKWID_DIR}/lib -llikwid ")
    include_directories(${LIKWID_DIR}/include)
endif()
