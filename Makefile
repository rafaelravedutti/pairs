.PHONY: all build clean lj_ns_cpu lj_ns_gpu

all: build lj_ns_cpu lj_ns_gpu
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	python3 setup.py build && python3 setup.py install --user

lj_ns_cpu:
	@echo "Generating and compiling Lennard-Jones example for CPU..."
	python3 examples/lj_func.py cpu

lj_ns_gpu:
	@echo "Generating and compiling Lennard-Jones example for GPU..."
	python3 examples/lj_func.py gpu

# Targets
cpu: build lj_ns_cpu
	g++ -o lj_ns lj_ns.cpp

gpu: build lj_ns_gpu
	nvcc -o lj_ns lj_ns.cu

clean:
	@echo "Cleaning..."
	rm -rf build lj_ns lj_ns.cpp lj_ns.cu dist pairs.egg-info functions functions.pdf
