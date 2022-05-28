#ifndef ISING_H
#define ISING_H

#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <iomanip>

#include <cuda_fp16.h>
#include <curand.h>
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "ising.cuh"

#define THREADS 256

#endif