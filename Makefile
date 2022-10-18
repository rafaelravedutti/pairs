.PHONY: all build clean

all: clean build lj_cpu lj_gpu
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	python3 setup.py build && python3 setup.py install --user

lj_ns.cpp:
	@echo "Generating and compiling Lennard-Jones example for CPU..."
	python3 examples/lj_func.py cpu

lj_ns.cu:
	@echo "Generating and compiling Lennard-Jones example for GPU..."
	python3 examples/lj_func.py gpu

# Targets
lj_cpu: lj_ns.cpp
#	g++ -o lj_cpu lj_ns.cpp -DDEBUG
	mpic++ -O3 -o lj_cpu lj_ns.cpp runtime/pairs.cpp runtime/domain/regular_6d_stencil.cpp runtime/devices/dummy.cpp -DDEBUG

lj_gpu: lj_ns.cu
	mpic++ -c -o pairs.o runtime/pairs.cpp -DDEBUG
	mpic++ -c -o regular_6d_stencil.o runtime/domain/regular_6d_stencil.cpp -DDEBUG
	nvcc -o lj_gpu runtime/devices/cuda.cu runtime.o regular_6d_stencil.o lj_ns.cu

clean:
	@echo "Cleaning..."
	rm -rf build lj_cpu lj_gpu lj_ns.cpp lj_ns.cu dist pairs.egg-info functions functions.pdf pairs.o regular_6d_stencil.o
