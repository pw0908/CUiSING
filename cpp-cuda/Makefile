CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE = $(CUDA_HOME)/include
CC=g++
LD=$(CUDACC)
LDFLAGS=-L$(CUDA_INCLUDE) -lcurand
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS= -std=c++17 -c -O3 -lineinfo -gencode arch=compute_86,code=sm_86
DIR_SRC = src
CUDA_FILES = $(wildcard ${DIR_SRC}/*.cu)
SRC=$(CUDA_FILES)
OBJ=$(SRC:.cu=.o)

TARGET = Ising

defaut: $(TARGET)
	-rm $(OBJ)

$(TARGET): $(OBJ)
	$(LD) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cu
	$(LD) $(CUDACFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm $(TARGET)