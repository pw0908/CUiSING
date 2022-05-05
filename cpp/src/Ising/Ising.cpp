#include "Ising.h"

/* Default destructor */
Ising::~Ising(){}

Ising::Ising()
{
    n_iters = 1000;
    n = 50;
    J = 1.0;
    h = 0.0;
    
    E.resize(n_iters+1);
    M.resize(n_iters+1);

    iter = 0;
}

/* Constructor */
Ising::Ising(int n_iters_, int n_, double J_, double h_)
{
    n = n_;
    J = J_;
    h = h_;
    n_iters = n_iters_;

    E.resize(n_iters+1);
    M.resize(n_iters+1);

    iter = 0;
}


void Ising::monteCarlo()
{
    initializeSystem();

    while (iter < n_iters)
    {
        getRandom();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                calcDeltaHamiltonian(i,j);
                if (dE <= 0 || R[i][j] <= exp(-1.0*dE))
                {
                    A[i][j] *= -1;
                }
            }
        }

        iter++;
        calcHamiltonian();
        calcMagnetization();
    }

    std::cout << "E = " << E[iter] << std::endl;
    std::cout << "M = " << M[iter] << std::endl;

    finalizeSystem();
}

void Ising::initializeSystem()
{
    N = n*n;

    A = new int *[n];
    R = new float *[n];

    for (int i = 0; i < n; i++)
    {
        A[i] = new int[n];
        R[i] = new float[n];
    }

    srand(time(NULL));
    /* Initialize spins on lattice */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = std::rand()%2 * 2 - 1;
        }
    }

    calcHamiltonian();
    calcMagnetization();
}

void Ising::finalizeSystem()
{

    for (int i = 0; i < n; i++)
    {
        delete[] A[i];
        delete[] R[i];
    }

}
