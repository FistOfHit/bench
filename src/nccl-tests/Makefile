##### NCCL-Tests Makefile (bundled minimal version) #####
# Based on https://github.com/NVIDIA/nccl-tests
# Builds all_reduce_perf, all_gather_perf, broadcast_perf,
# reduce_scatter_perf, reduce_perf

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
MPI ?= 0
MPI_HOME ?= /usr

CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
NCCL_INC  := $(NCCL_HOME)/include
NCCL_LIB  := $(NCCL_HOME)/lib

NVCC      := $(CUDA_HOME)/bin/nvcc
CXX       := g++

NVCC_GENCODE ?= -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_90,code=sm_90

CXXFLAGS  := -O3 -std=c++11
NVCCFLAGS := -O3 -std=c++11 $(NVCC_GENCODE)
LDFLAGS   := -L$(CUDA_LIB) -L$(NCCL_LIB) -lcudart -lnccl -lrt

ifeq ($(MPI), 1)
  CXXFLAGS  += -DMPI_SUPPORT -I$(MPI_HOME)/include
  NVCCFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
  LDFLAGS   += -L$(MPI_HOME)/lib -lmpi
endif

INCLUDES  := -I$(CUDA_INC) -I$(NCCL_INC) -Isrc

BUILDDIR  := build
SRCDIR    := src

TESTS := all_reduce_perf all_gather_perf broadcast_perf reduce_scatter_perf reduce_perf

all: $(BUILDDIR) $(addprefix $(BUILDDIR)/,$(TESTS))

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%_perf: $(SRCDIR)/%.cu $(SRCDIR)/common.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
