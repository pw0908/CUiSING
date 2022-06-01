#include "ising.h"

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
    CHECK_CUDA(cudaMalloc(&rands, round(std::pow(n,d)) * sizeof(*rands)));

    /* Generate initial random numbers on device, for initializing lattice */
    gen_rands(cg,rands,n,d);

    /* Setup lattice and allocate memory for storage */
    signed int *lattice;
    CHECK_CUDA(cudaMalloc(&lattice, round(std::pow(n,d)) * sizeof(*lattice)));
    /* Initialize the lattice using the previously generated random array on device */
    int blocks = (round(std::pow(n,d))+THREADS - 1)/THREADS;
    init_lattice<<<blocks,THREADS>>>(lattice,rands,n,d);

    /* Make sure the device is synced */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* initialize timer for monte carlo loop */
    auto t0 = std::chrono::high_resolution_clock::now();

    /* Enter monte carlo loop for n_iters iterations */
    if (d==2)
    {
        for (int i = 0; i < n_iters; i++)
        {
            callMcIteration2d(lattice,cg,rands,n,J,h);  // update the lattice
            callCalcHamiltonian2d(lattice,E,n,J,h,i);   // calculate the energy and store in E
            callCalcMagnetization2d(lattice,M,n,J,h,i); // calculate the mag and store in M
        }
    }
    else if (d==3)
    {
        for (int i = 0; i < n_iters; i++)
        {
            callMcIteration3d(lattice,cg,rands,n,J,h);  // update the lattice
            callCalcHamiltonian3d(lattice,E,n,J,h,i);   // calculate the energy and store in E
            callCalcMagnetization3d(lattice,M,n,J,h,i); // calculate the mag and store in M
        }
    }
    else
    {
        std::cerr << "Error: d = " << d << " is not a valid dimension (must be 2 or 3)\n";
        std::exit(0);
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
    lattice_h = new signed int [(long long)round(std::pow(n,d))];
    CHECK_CUDA(cudaMemcpy(lattice_h, lattice, round(std::pow(n,d))*sizeof(float), cudaMemcpyDeviceToHost));

    if (d==2)
    {
        print_lattice(lattice_h,n);
    }

    /* Output program duration */
    printf("Total Program Time : %f seconds\n", duration * 1e-6);

    
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