CUDA_HOME=/usr/local/cuda-11.4
CUDACC=$(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE = $(CUDA_HOME)/include
CC=g++
LD=$(CUDACC)
LDFLAGS=-L$(CUDA_INCLUDE) -lcurand
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS= -std=c++17 -c -O3 -lineinfo -arch=sm_8

all: ising
	rm main.o

ising: main.o
	$(LD) -o ising main.o $(LDFLAGS)

%.o: %.cu
	nvcc -c $(CUDACFLAGS) $<

.PHONY: clean
clean:
	rm ising

