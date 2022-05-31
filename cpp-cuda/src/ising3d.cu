#include "ising.h"

/**************************/
/*      CUDA Kernels      */
/**************************/

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
 * converted from linear indexing to 3d indexing, and then the mod
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
                                        const float h) {

    const long long tid = (long long)(blockDim.x)*blockIdx.x + threadIdx.x;
    const int i = tid/(n*n);
    const int j = (tid/n)%n;
    const int k = tid%n;
    
    if (tid >= n*n*n)
    {
        return;
    }
    else if ( ((k%2 == j%2) != i%2) != sub_lattice)
    {
        return;
    }
    
    const int sl = lattice[i*n*n+(n+(j-1)%n)%n*n+k];
    const int sr = lattice[i*n*n+(n+(j+1)%n)%n*n+k];
    const int su = lattice[(n+(i+1)%n)%n*n*n+j*n+k];
    const int sd = lattice[(n+(i-1)%n)%n*n*n+j*n+k];
    const int sn = lattice[i*n*n+j*n+(n+(k-1)%n)%n];
    const int ss = lattice[i*n*n+j*n+(n+(k+1)%n)%n];

    const int sum_spins = sl + sr + su + sd + sn + ss;
    const int sijk = lattice[tid];
    float boltz = exp(-2.0*sijk*(sum_spins*J+h));
    if (rands[tid]<=boltz) lattice[tid] = -sijk;
}


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
__global__ void cudaCalcHamiltonian3dKernel(signed int *lattice,
                                            float *E,
                                            const int n,
                                            const float J,
                                            const float h,
                                            const int iter) {
    
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    unsigned idx = tid + blockIdx.x * blockDim.x;
    shmem[tid] = 0.0;

    unsigned i,j,k;
    int sl,sr,su,sd,sn,ss,sijk;
    for (; idx < n*n*n; idx += blockDim.x * gridDim.x)
    {
        i = idx/(n*n);
        j = (idx/n)%n;
        k = idx%n;

        sl = lattice[i*n*n+(n+(j-1)%n)%n*n+k];
        sr = lattice[i*n*n+(n+(j+1)%n)%n*n+k];
        su = lattice[(n+(i+1)%n)%n*n*n+j*n+k];
        sd = lattice[(n+(i-1)%n)%n*n*n+j*n+k];
        sn = lattice[i*n*n+j*n+(n+(k-1)%n)%n];
        ss = lattice[i*n*n+j*n+(n+(k+1)%n)%n];
        sijk = lattice[idx];
        
        shmem[tid] -= sijk*(J*(sl+sr+su+sd+sn+ss)/2.0+h);
    }
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s>0; s>>=1)
    {
        if (tid < s)
            shmem[tid] += shmem[tid + s];
        
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&E[iter], shmem[0]/(3*n*n*n));
}


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
                                              const int iter) {
    
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    unsigned idx = tid + blockIdx.x * blockDim.x;
    shmem[tid] = 0.0;
    for (; idx < n*n*n; idx += blockDim.x * gridDim.x)
    {
        shmem[tid] += lattice[idx];
    }
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s>0; s>>=1)
    {
        if (tid < s)
            shmem[tid] += shmem[tid + s];
        
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&M[iter], shmem[0]/(n*n*n));
}

/**************************/
/*  Helper C++ Functions  */
/**************************/


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
void callMcIteration3d(signed int *lattice,
                       curandGenerator_t cg,
                       float *rands,
                       const int n,
                       const float J,
                       const float h){
    
    int blocks = (n*n*n+THREADS - 1)/THREADS;

    CHECK_CURAND(curandGenerateUniform(cg,rands,n*n*n));
    cudaMcIteration3dKernel<0><<<blocks,THREADS>>>(lattice,rands,n,J,h);
    cudaMcIteration3dKernel<1><<<blocks,THREADS>>>(lattice,rands,n,J,h);
}

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
                           const int iter) {

    int blocks = (n*n*n+THREADS - 1)/THREADS;
    cudaCalcHamiltonian3dKernel<<<blocks,THREADS>>>(lattice,E,n,J,h,iter);
}


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
void callCalcMagnetization3d(signed int *lattice,
                             float *M,
                             const int n,
                             const float J,
                             const float h,
                             const int iter) {

    int blocks = (n*n*n+THREADS - 1)/THREADS;
    cudaCalcMagnetization3dKernel<<<blocks,THREADS>>>(lattice,M,n,J,h,iter);
}

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
// void print_lattice(signed int *lattice, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             if (lattice[(i*n+j)] == 1)
//             {
//                 std::cout << " " << lattice[(i*n+j)] << " ";
//             }
//             else
//             {
//                 std::cout << lattice[(i*n+j)] << " ";
//             }
            
//         }
//         std::cout << " \n";
//     }
// }