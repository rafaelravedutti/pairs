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
	mpic++ -O3 -o lj_cpu lj_ns.cpp -DDEBUG

lj_gpu: lj_ns.cu
	nvcc -o lj_gpu lj_ns.cu

clean:
	@echo "Cleaning..."
	rm -rf build lj_cpu lj_gpu lj_ns.cpp lj_ns.cu dist pairs.egg-info functions functions.pdf
