.PHONY: all build clean lj_ns

all: build lj_ns clean
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	python3 setup.py build && python3 setup.py install --user

lj_ns:
	@echo "Generating and compiling CPP for Lennard-Jones example..."
	python3 examples/lj_func.py
	g++ -o lj_ns lj_ns.cpp

clean:
	@echo "Cleaning..."
	rm -rf build lj_ns lj_ns.cpp dist pairs.egg-info functions functions.pdf
