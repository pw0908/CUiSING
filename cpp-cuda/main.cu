#include "ising.h"

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
__global__ void init_lattice_2d(signed int *lattice,
                                const float* __restrict__ rands,
                                const long long n) {
    
    const long long tid = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (tid >= n*n) return;
    float r = rands[tid];
    lattice[tid] = (r < 0.5f) ? -1 : 1;
}

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
                                        const float h) {

    const long long tid = (long long)(blockDim.x)*blockIdx.x + threadIdx.x;
    const int i = tid / n;
    const int j = tid % n;
    
    if (tid >= n*n)
    {
        return;
    }
    else if ( (i%2 != j%2) != sub_lattice)
    {
        return;
    }

    const int sl = lattice[i*n+(n+(j-1)%n)%n];
    const int sr = lattice[i*n+(n+(j+1)%n)%n];
    const int su = lattice[(n+(i+1)%n)%n*n+j];
    const int sd = lattice[(n+(i-1)%n)%n*n+j];

    const int sum_spins = sl + sr + su + sd;
    const int sij = lattice[tid];
    float boltz = exp(-2.0*sij*(sum_spins*J+h));
    if (rands[tid]<=boltz) lattice[tid] = -sij;
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
__global__ void cudaCalcHamiltonian2dKernel(signed int *lattice,
                                            float *E,
                                            const int n,
                                            const float J,
                                            const float h,
                                            const int iter) {
    
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    unsigned idx = tid + blockIdx.x * blockDim.x;
    shmem[tid] = 0.0;
    for (; idx < n*n; idx += blockDim.x * gridDim.x)
    {
        unsigned i = idx / n;
        unsigned j = idx % n;
        int sl = lattice[i*n+(n+(j-1)%n)%n];
        int sr = lattice[i*n+(n+(j+1)%n)%n];
        int su = lattice[(n+(i+1)%n)%n*n+j];
        int sd = lattice[(n+(i-1)%n)%n*n+j];
        int sij = lattice[idx];
        
        shmem[tid] -= sij*(J*(sl+sr+su+sd)/2.0+h);
    }
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s>0; s>>=1)
    {
        if (tid < s)
            shmem[tid] += shmem[tid + s];
        
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&E[iter], shmem[0]/(n*n));
}


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
                                              const int iter) {
    
    extern __shared__ float shmem[];

    unsigned tid = threadIdx.x;
    unsigned idx = tid + blockIdx.x * blockDim.x;
    shmem[tid] = 0.0;
    for (; idx < n*n; idx += blockDim.x * gridDim.x)
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
        atomicAdd(&M[iter], shmem[0]/(n*n));
}

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
void gen_rands(curandGenerator_t cg, float *rands, int n)
{
    CHECK_CURAND(curandGenerateUniform(cg,rands,n*n));
}


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
                       const float h){
    
    int blocks = (n*n+THREADS - 1)/THREADS;

    CHECK_CURAND(curandGenerateUniform(cg,rands,n*n));
    cudaMcIteration2dKernel<0><<<blocks,THREADS>>>(lattice,rands,n,J,h);
    cudaMcIteration2dKernel<1><<<blocks,THREADS>>>(lattice,rands,n,J,h);
}

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
                           const int iter) {

    int blocks = (n*n+THREADS - 1)/THREADS;
    cudaCalcHamiltonian2dKernel<<<blocks,THREADS>>>(lattice,E,n,J,h,iter);
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
void callCalcMagnetization2d(signed int *lattice,
                             float *M,
                             const int n,
                             const float J,
                             const float h,
                             const int iter) {

    int blocks = (n*n+THREADS - 1)/THREADS;
    cudaCalcMagnetization2dKernel<<<blocks,THREADS>>>(lattice,M,n,J,h,iter);
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
void print_lattice(signed int *lattice, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (lattice[(i*n+j)] == 1)
            {
                std::cout << " " << lattice[(i*n+j)] << " ";
            }
            else
            {
                std::cout << lattice[(i*n+j)] << " ";
            }
            
        }
        std::cout << " \n";
    }
}

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
void writeEM(float *E_h, float *M_h, const int n_iters)
{
    std::ofstream emStream;
    std::string emFile = "EM.dat";
    emStream.open(emFile.c_str(), std::fstream::out | std::ofstream::trunc);

    for (unsigned i = 0; i < n_iters; i++)
    {
        if (!(i%10))
        {
            emStream << i << " ";
            emStream << std::fixed << std::setprecision(5) << E_h[i] << " ";
            emStream << std::fixed << std::setprecision(5) << M_h[i] << " \n";
        }
        
    }

}


/* Alas: the main event */
int main(int argc, char **argv){


    /* Default System Parameters */
    unsigned n_iters = 1000;
    unsigned d = 2;
    unsigned n = 100;
    double J = 1.0;
    double h = 0.0;
    
    /* Parse input arguments from the command line */
    if (argc > 5)
    {
        n_iters = atof(argv[1]);
        d = atof(argv[2]);
        n = atof(argv[3]);
        J = atof(argv[4]);
        h = atof(argv[5]);
    }
    else if (argc > 1)
    {
        std::cerr << "Need to supply 5 arguments (or none): int n_iters, int n, double J, double h" << std::endl;
        std::exit(0);
    }

    /* Print a window with selected system parameters */
    std::cout << "=========================" << std::endl;
    std::cout << "n_iters:   " << n_iters    << std::endl;
    std::cout << "d:         " << d          << std::endl;
    std::cout << "n:         " << n          << std::endl;
    std::cout << "J:         " << J          << std::endl;
    std::cout << "h:         " << h          << std::endl;
    std::cout << "=========================" << std::endl;

    /* Create and allocate E and M trajectories on DEVICE */
    float *E;
    float *M;
    CHECK_CUDA(cudaMalloc(&E, n_iters * sizeof(*E)));
    CHECK_CUDA(cudaMalloc(&M, n_iters * sizeof(*M)));

    /* Create and allocate E_h and M_h trajectories on HOST */
    float *E_h;
    float *M_h;
    E_h = new float[n_iters];
    M_h = new float[n_iters];

    /* rng seeding, based on system time */
    time_t seed;
    time(&seed); // store system time in seed
    srand((unsigned int) seed); // seed random number generator with system time
    
    /* Setup cuRAND and allocated device memory for storing random numbers */
    curandGenerator_t cg;
    CHECK_CURAND(curandCreateGenerator(&cg, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(cg, seed));
    float *rands;
    CHECK_CUDA(cudaMalloc(&rands, n * n * sizeof(*rands)));

    /* Generate initial random numbers on device, for initializing lattice */
    gen_rands(cg,rands,n);

    /* Setup lattice and allocate memory for storage */
    signed int *lattice;
    CHECK_CUDA(cudaMalloc(&lattice, n * n * sizeof(*lattice)));
    /* Initialize the lattice using the previously generated random array on device */
    int blocks = (n*n+THREADS - 1)/THREADS;
    init_lattice_2d<<<blocks,THREADS>>>(lattice,rands, n);

    /* Make sure the device is synced */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* initialize timer for monte carlo loop */
    auto t0 = std::chrono::high_resolution_clock::now();

    /* Enter monte carlo loop for n_iters iterations */
    for (int i = 0; i < n_iters; i++)
    {
        callMcIteration2d(lattice,cg,rands,n,J,h);  // update the lattice
        callCalcHamiltonian2d(lattice,E,n,J,h,i);   // calculate the energy and store in E
        callCalcMagnetization2d(lattice,M,n,J,h,i); // calculate the mag and store in M
    }

    /* Ensure device is synced */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* End timer for monte carlo and calculate program duration */
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();

    /* Copy the E and M trajectory to host for writing */
    CHECK_CUDA(cudaMemcpy(E_h, E, n_iters*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(M_h, M, n_iters*sizeof(float), cudaMemcpyDeviceToHost));
    writeEM(E_h,M_h,n_iters); // write traj to file EM.dat

    /* Copy the final latice configuration to the host */
    signed int *lattice_h;
    lattice_h = new signed int [n*n];
    CHECK_CUDA(cudaMemcpy(lattice_h, lattice, n*n * sizeof(float), cudaMemcpyDeviceToHost));

    /* Output program duration */
    printf("Total Program Time: %f seconds\n", duration * 1e-6);

    
    /* Cleaning up */
    CHECK_CURAND(curandDestroyGenerator(cg));
    CHECK_CUDA(cudaFree(rands));
    CHECK_CUDA(cudaFree(lattice));
    CHECK_CUDA(cudaFree(E));
    CHECK_CUDA(cudaFree(M));
    delete[] lattice_h;
    delete[] E_h;
    delete[] M_h;

    return 0;
}
