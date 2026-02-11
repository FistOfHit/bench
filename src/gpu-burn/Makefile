##### gpu-burn Makefile (bundled version) V1.1 #####
# Based on https://github.com/wilicc/gpu-burn
# Fixed: PTX generation now uses single virtual architecture (portable)

CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

# Multi-arch for binary compilation
NVCC_FLAGS := -O3 -gencode=arch=compute_70,code=sm_70 \
              -gencode=arch=compute_80,code=sm_80 \
              -gencode=arch=compute_90,code=sm_90

# Single virtual arch for PTX (portable, forward-compatible)
NVCC_PTX_FLAGS := -O3 -arch=compute_70 -ptx

CXXFLAGS   := -O3
LDFLAGS    := -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -ldl

all: gpu_burn compare.ptx

gpu_burn: gpu_burn-drv.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDA_PATH)/include -o $@ $< $(LDFLAGS) -lpthread

compare.ptx: compare.cu
	$(NVCC) $(NVCC_PTX_FLAGS) -o $@ $<

clean:
	rm -f gpu_burn compare.ptx

.PHONY: all clean
