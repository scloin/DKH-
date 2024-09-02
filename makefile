CUDA_HOME ?= /usr/local/cuda
EX_HOME ?= /home/deepl/sooho/cudagraph_test
CODE_HOME ?= /home/deepl/sooho/cudagraph_test

all: main main_ device_r while

# test: frontier_thd_test no_frontier_test

# cpp: cpptest

# just: justgpu justgpu_test

main: $(CODE_HOME)/main.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/main.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/main -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11

main_: $(CODE_HOME)/main_.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/main_.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/main_ -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11

device: $(CODE_HOME)/device.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/device.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/device -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11

device_r: $(CODE_HOME)/device_r.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/device_r.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/device_r -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11

while: $(CODE_HOME)/while.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/while.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/while -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11


run_main: $(EX_HOME)/exec/main
	$(EX_HOME)/exec/main

run_imed:$(CODE_HOME)/main.cu $(CODE_HOME)/include/func.cu
	$(CUDA_HOME)/bin/nvcc -g -lineinfo $(CODE_HOME)/main.cu $(CODE_HOME)/include/func.cu -o $(EX_HOME)/exec/main -I$(CUDA_HOME)/include,$(CODE_HOME)/include,$(EX_HOME)/.. -L$(CUDA_HOME)/lib64 -std=c++11
	$(EX_HOME)/exec/main
