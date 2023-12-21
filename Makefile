.PHONY: all build clean

# General settings
TESTCASE=dem
PYCMD=python3

# C/C++ compiler settings
CC=mpicc
#CC=mpiicpx
#CC=mpiicpc
CFLAGS=-Ofast -march=core-avx2 -fopenmp ${MPI_FLAGS} ${LIKWID_FLAGS}
#CFLAGS=-Ofast -xHost -qopt-zmm-usage=high ${MPI_FLAGS} ${LIKWID_FLAGS}
#CFLAGS=-Ofast -xCORE-AVX512 -qopt-zmm-usage=high ${MPI_FLAGS} ${LIKWID_FLAGS}
DEBUG_FLAGS=
#DEBUG_FLAGS=-DDEBUG

# CUDA settings
NVCC=nvcc
NVCC_FLAGS=-O3 --use_fast_math
NVCC_PATH:="$(shell which ${NVCC})"
CUDA_FLAGS=-DENABLE_CUDA_AWARE_MPI
CUDART_FLAGS=-lcudart -L /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/cuda/12.2/targets/x86_64-linux/lib

# MPI settings
MPI_PATH=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/intel-2021.10.0/openmpi-4.1.6-ijsnjhq77rjc256wlrp52m37rsq6miff
MPI_FLAGS=-I${MPI_PATH}/include

# Likwid settings
LIKWID_INC ?= -I/usr/local/include
LIKWID_DEFINES ?= -DLIKWID_PERFMON
LIKWID_LIB ?= -L/usr/local/lib
LIKWID_FLAGS = -llikwid ${LIKWID_INC} ${LIKWID_DEFINES} ${LIKWID_LIB}

# Other
CPU_OBJ_PATH=obj_cpu
CPU_SRC="$(TESTCASE).cpp"
CPU_BIN="$(TESTCASE)_cpu"
GPU_OBJ_PATH=obj_gpu
GPU_SRC="$(TESTCASE).cu"
GPU_BIN="$(TESTCASE)_gpu"

all: clean build $(CPU_BIN) $(GPU_BIN)
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	$(PYCMD) setup.py build && $(PYCMD) setup.py install --user

$(CPU_SRC):
	@echo "Generating and compiling $(TESTCASE) example for CPU..."
	@mkdir -p $(CPU_OBJ_PATH)
	$(PYCMD) examples/$(TESTCASE).py cpu

$(GPU_SRC):
	@echo "Generating and compiling $(TESTCASE) example for GPU..."
	@mkdir -p $(GPU_OBJ_PATH)
	$(PYCMD) examples/$(TESTCASE).py gpu

$(CPU_OBJ_PATH)/pairs.o: runtime/pairs.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS) $(CFLAGS)

$(CPU_OBJ_PATH)/regular_6d_stencil.o: runtime/domain/regular_6d_stencil.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS) $(CFLAGS)

$(CPU_OBJ_PATH)/dummy.o: runtime/devices/dummy.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS) $(CFLAGS)

$(GPU_OBJ_PATH)/pairs.o: runtime/pairs.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(CUDA_FLAGS)

$(GPU_OBJ_PATH)/regular_6d_stencil.o: runtime/domain/regular_6d_stencil.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(CUDA_FLAGS)

$(GPU_OBJ_PATH)/cuda_runtime.o: runtime/devices/cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< $(DEBUG_FLAGS) $(MPI_FLAGS) $(CUDA_FLAGS)

# Targets
$(CPU_BIN): $(CPU_SRC) $(CPU_OBJ_PATH)/pairs.o $(CPU_OBJ_PATH)/regular_6d_stencil.o $(CPU_OBJ_PATH)/dummy.o
	$(CC) $(CFLAGS) -o $(CPU_BIN) $(CPU_SRC) $(CPU_OBJ_PATH)/pairs.o $(CPU_OBJ_PATH)/regular_6d_stencil.o $(CPU_OBJ_PATH)/dummy.o $(DEBUG_FLAGS)

$(GPU_BIN): $(GPU_SRC) $(GPU_OBJ_PATH)/pairs.o $(GPU_OBJ_PATH)/regular_6d_stencil.o $(GPU_OBJ_PATH)/cuda_runtime.o
	$(NVCC) $(NVCC_FLAGS) -c -o $(GPU_OBJ_PATH)/$(GPU_BIN).o $(GPU_SRC) $(DEBUG_FLAGS) $(MPI_FLAGS) $(CUDA_FLAGS)
	$(CC) -o $(GPU_BIN) $(GPU_OBJ_PATH)/$(GPU_BIN).o $(GPU_OBJ_PATH)/cuda_runtime.o $(GPU_OBJ_PATH)/pairs.o $(GPU_OBJ_PATH)/regular_6d_stencil.o $(CUDART_FLAGS) $(CUDA_FLAGS) $(CFLAGS)

clean:
	@echo "Cleaning..."
	rm -rf build $(CPU_BIN) $(GPU_BIN) lj.o $(CPU_SRC) $(GPU_SRC) dist pairs.egg-info functions functions.pdf $(CPU_OBJ_PATH) $(GPU_OBJ_PATH)
