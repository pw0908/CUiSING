#ifndef ISING3D_CUH
#define ISING3D_CUH

#include "cuda_helper.h"

/**************************/
/*      CUDA Kernels      */
/**************************/

/* !!!!!UNFINISHED!!!! */
/* __global__ void cudaMCIteration3dKernel
 *
 * Location: DEVICE
 *
 * This CUDA kernel is run on the device in order to run monte carlo 
 * updates in parallel. The kernel implements a checkerboard algorithm
 * to only attempts flips of non-interacting spins that can be flipped
 * in parallel. The attempted flips are accepted/rejected based on the
 * Metropolis criteria which uses the Boltzmann weight. The lattice is
 * updated directly on the device memory.
 * 
 * Periodic boundary conditions are considered. The thread id is
 * converted from linear indexing to 2d indexing, and then the mod
 * operator (%) is employed to wrap the lattice in both dimensions,
 * enforcing periodic boundary conditions.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - const float *__restrict__ rands: numbers generated from U(0,1), stored in device memory
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 * 
 * Output:
 *    - none (void)
 * 
 * */
template<int sub_lattice>
__global__ void cudaMcIteration3dKernel(signed int *lattice,
                                        const float *__restrict__ rands,
                                        const int n,
                                        const float J,
                                        const float h);


/* __global__ void cudaCalcHamiltonian3dKernel
 *
 * Location: DEVICE
 * 
 * This CUDA kernel is run on the device in order to calculate the
 * total Hamiltonian (energy) rapidly. The kernel utilizes shared
 * memory and a binary recution, followed by an atomic addition
 * to maximize the speed of the sum. The local energy of each spin
 * is calculated per thread.
 * 
 * Periodic boundary conditions are considered. The thread id is
 * converted from linear indexing to 2d indexing, and then the mod
 * operator (%) is employed to wrap the lattice in both dimensions,
 * enforcing periodic boundary conditions.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - float *E: energy trajectory stored in device memory
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 *    - const int iter: the current monte carlo iteration
 * 
 * Output:
 *    - none (void)
 * 
 * */
__global__ void cudaCalcHamiltonian3dKernel(signed int *lattice,
                                            float *E,
                                            const int n,
                                            const float J,
                                            const float h,
                                            const int iter);


/* __global__ void cudaCalcMagnetization3dKernel
 *
 * Location: DEVICE
 * 
 * This CUDA kernel is run on the device in order to calculate the
 * total relative magnetization rapidly. The kernel utilizes shared
 * memory and a binary recution, followed by an atomic addition
 * to maximize the speed of the sum. Relative magnetization is the
 * sum of all spin values divided by the total number of spins.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - float *M: magnetization trajectory stored in device memory
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 *    - const int iter: the current monte carlo iteration
 * 
 * Output:
 *    - none (void)
 * 
 * */
__global__ void cudaCalcMagnetization3dKernel(signed int *lattice,
                                              float *M,
                                              const int n,
                                              const float J,
                                              const float h,
                                              const int iter);

/**************************/
/*  Helper C++ Functions  */
/**************************/

/* void callMCIteration3d
 *
 * Location: HOST
 * 
 * This host function makes a call to cudaMcIteration2dKernel
 * which is the kernel that runs a single monte carlo iteration
 * on the provided lattice with the provided system parameters,
 * and updates the lattice accordingly, all on device memory.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - curandGenerator_t cg: a cuRAND generator
 *    - float *rands: array for randodm numbers stored on device
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 * 
 * Output:
 *    - none (void)
 * 
 * */
void callMcIteration3d(signed int *lattice,
                       curandGenerator_t cg,
                       float *rands,
                       const int n,
                       const float J,
                       const float h);

/* void callCalcHamiltonian3d
 *
 * Location: HOST
 * 
 * This host function makes a call to callCalcHamiltonian2d
 * which is the kernel that takes the lattice configuration
 * and calculates the total Hamiltonion (energy) on the device.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - float *E: energy trajectory stored in device memory
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 *    - const int iter: the current monte carlo iteration
 * 
 * Output:
 *    - none (void)
 * 
 * */
void callCalcHamiltonian3d(signed int *lattice,
                           float *E,
                           const int n,
                           const float J,
                           const float h,
                           const int iter);


/* void callCalcMagnetization3d
 *
 * Location: HOST
 * 
 * This host function makes a call to callCalcMagnetization2d
 * which is the kernel that takes the lattice configuration
 * and calculates the relative magnetization on the device.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - float *M: magnetization trajectory stored in device memory
 *    - const int n: number of spins in each dimension
 *    - const float J: the spin-spin interaction strength
 *    - const float h: the spin-field interaction strength
 *    - const int iter: the current monte carlo iteration
 * 
 * Output:
 *    - none (void)
 * 
 * */
void callCalcMagnetization3d(signed int *lattice,
                             float *M,
                             const int n,
                             const float J,
                             const float h,
                             const int iter);

/* void print_lattice_3d
 *
 * Location: HOST
 * 
 * Function to print the lattice configuration from host memory,
 * for debugging purposes mostly.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in HOST memeory
 *    - const int n: number of spins in each dimension
 * 
 * Output:
 *    - none (void)
 * 
 * */
// void print_lattice_3d(signed int *lattice, int n);

#endif