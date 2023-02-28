.PHONY: all build clean

PYCMD=python3
CC=mpicxx
NVCC=nvcc
NVCC_PATH = "$(shell which ${NVCC})"
CUDA_BIN_PATH = "$(shell dirname ${NVCC_PATH})"
CUDA_PATH:="$(shell dirname ${CUDA_BIN_PATH})"

all: clean build lj_cpu lj_gpu
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	$(PYCMD) setup.py build && $(PYCMD) setup.py install --user

lj_ns.cpp:
	@echo "Generating and compiling Lennard-Jones example for CPU..."
	$(PYCMD) examples/lj_func.py cpu

lj_ns.cu:
	@echo "Generating and compiling Lennard-Jones example for GPU..."
	$(PYCMD) examples/lj_func.py gpu

# Targets
lj_cpu: lj_ns.cpp
#	g++ -o lj_cpu lj_ns.cpp -DDEBUG
	$(CC) -O3 -o lj_cpu lj_ns.cpp runtime/pairs.cpp runtime/domain/regular_6d_stencil.cpp runtime/devices/dummy.cpp -DDEBUG

lj_gpu: lj_ns.cu
	$(CC) -c -o pairs.o runtime/pairs.cpp -DDEBUG
	$(CC) -c -o regular_6d_stencil.o runtime/domain/regular_6d_stencil.cpp -DDEBUG
	$(NVCC) -c -o cuda_runtime.o runtime/devices/cuda.cu -DDEBUG
	$(NVCC) -c -o lj_gpu.o lj_ns.cu -DDEBUG
	$(CC) -o lj_gpu lj_gpu.o cuda_runtime.o pairs.o regular_6d_stencil.o -lcudart -L$(CUDA_PATH)/lib64

clean:
	@echo "Cleaning..."
	rm -rf build lj_cpu lj_gpu lj_ns.cpp lj_ns.cu dist pairs.egg-info functions functions.pdf pairs.o regular_6d_stencil.o cuda_runtime.o lj_gpu.o
