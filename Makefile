.PHONY: all build clean lj_ns

all: build lj_ns
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	python3 setup.py build && python3 setup.py install --user

lj_ns:
	@echo "Generating and compiling CPP for Lennard-Jones example..."
	python3 examples/lj_func.py

# Targets
cpu: build lj_ns
	g++ -o lj_ns lj_ns.cpp

gpu: build lj_ns
	nvcc -o lj_ns lj_ns.cu

clean:
	@echo "Cleaning..."
	rm -rf build lj_ns lj_ns.cpp dist pairs.egg-info functions functions.pdf
