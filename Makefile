.PHONY: all build clean

TESTCASE=lj
PYCMD=python3
CC=mpicxx
NVCC=nvcc
NVCC_PATH:="$(shell which ${NVCC})"
CUDA_BIN_PATH:="$(shell dirname ${NVCC_PATH})"
CUDA_PATH:="$(shell dirname ${CUDA_BIN_PATH})"
OBJ_PATH=obj
CPU_SRC="$(TESTCASE).cpp"
CPU_BIN="$(TESTCASE)_cpu"
GPU_SRC="$(TESTCASE).cu"
GPU_BIN="$(TESTCASE)_gpu"
DEBUG_FLAGS="-DDEBUG"

all: clean build $(CPU_BIN) $(GPU_BIN)
	@echo "Everything was done!"

build:
	@echo "Building pairs package..."
	$(PYCMD) setup.py build && $(PYCMD) setup.py install --user

$(CPU_SRC):
	@echo "Generating and compiling $(TESTCASE) example for CPU..."
	@mkdir -p $(OBJ_PATH)
	$(PYCMD) examples/$(TESTCASE).py cpu

$(GPU_SRC):
	@echo "Generating and compiling $(TESTCASE) example for GPU..."
	@mkdir -p $(OBJ_PATH)
	$(PYCMD) examples/$(TESTCASE).py gpu

$(OBJ_PATH)/pairs.o: runtime/pairs.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS)

$(OBJ_PATH)/regular_6d_stencil.o: runtime/domain/regular_6d_stencil.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS)

$(OBJ_PATH)/dummy.o: runtime/devices/dummy.cpp
	$(CC) -c -o $@ $< $(DEBUG_FLAGS)

# Targets
$(CPU_BIN): $(CPU_SRC) $(OBJ_PATH)/pairs.o $(OBJ_PATH)/regular_6d_stencil.o $(OBJ_PATH)/dummy.o
	$(CC) -O3 -o $(CPU_BIN) $(CPU_SRC) $(OBJ_PATH)/pairs.o $(OBJ_PATH)/regular_6d_stencil.o $(OBJ_PATH)/dummy.o -DDEBUG

$(GPU_BIN): $(GPU_SRC) $(OBJ_PATH)/pairs.o $(OBJ_PATH)/regular_6d_stencil.o 
	$(NVCC) -c -o $(OBJ_PATH)/cuda_runtime.o runtime/devices/cuda.cu -DDEBUG
	$(NVCC) -c -o $(OBJ_PATH)/$(GPU_BIN).o $(GPU_SRC) -DDEBUG
	$(CC) -o $(GPU_BIN) $(OBJ_PATH)/$(GPU_BIN).o $(OBJ_PATH)/cuda_runtime.o $(OBJ_PATH)/pairs.o $(OBJ_PATH)/regular_6d_stencil.o -lcudart -L$(CUDA_PATH)/lib64

clean:
	@echo "Cleaning..."
	rm -rf build $(CPU_BIN) $(GPU_BIN) lj.o $(CPU_SRC) $(GPU_SRC) dist pairs.egg-info functions functions.pdf $(OBJ_PATH)/pairs.o $(OBJ_PATH)/regular_6d_stencil.o $(OBJ_PATH)/cuda_runtime.o $(OBJ_PATH)/$(GPU_BIN).o
