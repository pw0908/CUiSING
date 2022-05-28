CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE = $(CUDA_HOME)/include
CC=g++
LD=$(CUDACC)
LDFLAGS=-L$(CUDA_INCLUDE) -lcurand
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS= -std=c++17 -c -O3 -lineinfo -arch=sm_86
DIR_SRC= src
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
	rm $(TARGET) $(OBJ)









# # CS 179 Lab 4 Unix Makefile
# # Written by Loko Kung, 2018
# # Edited by Tyler Port, 2018
# # Edited by George Stathopoulos, 2019

# # Input Names
# CUDA_FILES = ising.cu main.cu
# CPP_FILES = 

# SRCDIR = src
# OBJDIR = obj

# # ------------------------------------------------------------------------------

# # CUDA Compiler and Flags
# CUDA_PATH = /usr/local/cuda-11.4
# CUDA_INC_PATH = $(CUDA_PATH)/include
# CUDA_BIN_PATH = $(CUDA_PATH)/bin
# CUDA_LIB_PATH = $(CUDA_PATH)/lib64
# # CUDA_LIB_PATH = $(CUDA_PATH)/lib

# NVCC = $(CUDA_BIN_PATH)/nvcc

# # OS-architecture specific flags
# ifeq ($(OS_SIZE),32)
# NVCC_FLAGS := -m32
# else
# NVCC_FLAGS := -m64
# endif
# NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++17 \
#              --expt-relaxed-constexpr
# NVCC_INCLUDE =
# NVCC_LIBS = 
# # NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
# # 		-gencode arch=compute_35,code=sm_35 \
# # 		-gencode arch=compute_50,code=sm_50 \
# # 		-gencode arch=compute_52,code=sm_52 \
# # 		-gencode arch=compute_60,code=sm_60 \
# # 		-gencode arch=compute_61,code=sm_61 \
# # 		-gencode arch=compute_61,code=compute_61
# NVCC_GENCODES = -gencode=arch=compute_80,code=sm_80 \
# 		-gencode=arch=compute_86,code=sm_86 \
# 		-gencode=arch=compute_87,code=sm_87 \
# 		-gencode=arch=compute_87,code=compute_87

# # CUDA Object Files
# CUDA_OBJ = $(OBJDIR)/cuda.o
# CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDA_FILES)))
# COMMON_OBJ = $(CUDA_OBJ) $(CUDA_OBJ_FILES)

# # ------------------------------------------------------------------------------

# # CUDA Linker and Flags
# CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# # ------------------------------------------------------------------------------

# # C++ Compiler and Flags
# GPP = g++
# FLAGS = -g -Wall -D_REENTRANT -std=c++17 -pthread
# INCLUDE = -I$(CUDA_INC_PATH) 
# LIBS = -L$(CUDA_INC_PATH) -lcurand

# # ------------------------------------------------------------------------------
# # Make Rules (Lab 3 specific)
# # ------------------------------------------------------------------------------

# # C++ Object Files
# OBJ_CPU = $(addprefix cpu-, $(notdir $(addsuffix .o, $(CPP_FILES))))
# OBJ=$(wildcard ./*.o)

# # Top level rules
# all: Ising

# Ising: $(OBJ_CPU) $(COMMON_OBJ)
# 	$(GPP) $(FLAGS) -o $@ $(INCLUDE) $^ $(LIBS) 

# # Compile C++ Source Files
# # cpu-%.cpp.o: $(addprefix $(SRCDIR)/, $(CPP_FILES))
# # 	$(GPP) $(FLAGS) -D CPU_ON=1 -c -o $@ $(INCLUDE) $< 

# # Compile CUDA Source Files
# %.cu.o: $(SRCDIR)/%.cu
# 	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# # Compile CUDA Source Files
# $(CUDA_OBJ_FILES): $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
# 	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# # Make linked device code
# $(CUDA_OBJ): $(CUDA_OBJ_FILES)
# 	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $<


# .PHONY: clean

# # Clean everything including temporary Emacs files
# clean:
# 	rm -f Ising $(OBJDIR)/*.o $(SRCDIR)/*~ *~


