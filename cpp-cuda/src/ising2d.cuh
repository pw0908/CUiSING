#ifndef ISING2D_CUH
#define ISING2D_CUH

#include "cuda_helper.h"

/**************************/
/*      CUDA Kernels      */
/**************************/

/* __global__ void init_lattice_2d
 *
 * Location: DEVICE
 * 
 * This CUDA kernel is run on the device in order to fill the lattice
 * with up and down spins. It takes in an array of random uniformly
 * distributed numbers and fills the lattice with integers, either 1
 * or -1. The filling is done in parallel.
 * 
 * Inputs:
 *    - signed int *lattice: the lattice stored in device memory
 *    - const float *__restrict__ rands: numbers generated from U(0,1), stored in device memory
 *    - const long long n: number of spins in each dimension
 * 
 * Outputs:
 *    - none (void)
 * 
 */
__global__ void init_lattice(signed int *lattice,
                                const float* __restrict__ rands,
                                const long long n,
                                const int d);



/* !!!!!UNFINISHED!!!! */
/* __global__ void cudaMCIteration2dKernel
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
__global__ void cudaMcIteration2dKernel(signed int *lattice,
                                        const float *__restrict__ rands,
                                        const int n,
                                        const float J,
                                        const float h);


/* __global__ void cudaCalcHamiltonian2dKernel
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
__global__ void cudaCalcHamiltonian2dKernel(signed int *lattice,
                                            float *E,
                                            const int n,
                                            const float J,
                                            const float h,
                                            const int iter);


/* __global__ void cudaCalcMagnetization2dKernel
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
__global__ void cudaCalcMagnetization2dKernel(signed int *lattice,
                                              float *M,
                                              const int n,
                                              const float J,
                                              const float h,
                                              const int iter);

/**************************/
/*  Helper C++ Functions  */
/**************************/

/* void gen_rands
 *
 * Location: HOST
 * 
 * This host function makes a call to cuRAND to generate pseudorandom
 * numbers drawn from U(0,1) on the device.
 * 
 * Inputs:
 *    - curandGenerator_t cg: a cuRAND generator
 *    - float *rands: array for randodm numbers stored on device
 *    - const int n: number of spins in each dimension
 * 
 * Output:
 *    - none (void)
 * 
 * */
void gen_rands(curandGenerator_t cg, float *rands, const int n, const int d);


/* void callMCIteration2d
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
void callMcIteration2d(signed int *lattice,
                       curandGenerator_t cg,
                       float *rands,
                       const int n,
                       const float J,
                       const float h);

/* void callCalcHamiltonian2d
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
void callCalcHamiltonian2d(signed int *lattice,
                           float *E,
                           const int n,
                           const float J,
                           const float h,
                           const int iter);


/* void callCalcMagnetization2d
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
void callCalcMagnetization2d(signed int *lattice,
                             float *M,
                             const int n,
                             const float J,
                             const float h,
                             const int iter);

/* void print_lattice
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
void print_lattice(signed int *lattice, int n);

/* void writeEM
 *
 * Location: HOST
 * 
 * Function to write the energy (E) and magnetization (M) trajectories
 * to a file in the root directoy, called 'EM.dat'.
 * 
 * Inputs:
 *    - float *E_h: the energy trajectory stored on host memory
 *    - float *M_h: the magnetization trajectory stored on host memory
 *    - const int n_iters: the number of MC iterations
 * 
 * Output:
 *    - none (void)
 * 
 * */
void writeEM(float *E_h, float *M_h, const int n_iters);

#endif