#include "Ising.h"

/* Default destructor */
Ising::~Ising(){}

Ising::Ising()
{
    n_iters = 1000;
    d = 2;
    n = 50;
    J = 1.0;
    h = 0.0;
    sample_freq = 10;
    
    E.resize(n_iters+1);
    M.resize(n_iters+1);

    iter = 0;

    setOutputFileStructure();
}

/* Constructor */
Ising::Ising(int n_iters_, int d_, int n_, double J_, double h_)
{
    n_iters = n_iters_;
    d = d_;
    n = n_;
    J = J_;
    h = h_;
    sample_freq = 10;
    
    E.resize(n_iters+1);
    M.resize(n_iters+1);

    iter = 0;

    setOutputFileStructure();
}

/*
* This loop can be semi-parallelized since spins only interact with
* their nearest neighbors. We can implement a checkerboard stencil
* where we flip spins in parallel so long as they are non-interacting
* which is totally legal. Instead of all spins being calculated in
* series, we can flip up to half of the spins in parallel utilizing the
* many threads on the GPU.
*/
void Ising::monteCarlo2d()
{
    initializeSystem2d();

    while (iter < n_iters)
    {
        getRandom2d();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                calcDeltaHamiltonian2d(i,j);
                if (dE <= 0 || R2d[i][j] <= exp(-1.0*dE))
                {
                    A2d[i][j] *= -1;
                }
            }
        }

        iter++;
        calcHamiltonian2d();
        calcMagnetization2d();
    }

    // std::cout << "E = " << E[iter] << std::endl;
    // std::cout << "M = " << M[iter] << std::endl;

    // writeOutput();
    finalizeSystem2d();
}

/*
* This loop can be semi-parallelized since spins only interact with
* their nearest neighbors. We can implement a checkerboard stencil
* where we flip spins in parallel so long as they are non-interacting
* which is totally legal. Instead of all spins being calculated in
* series, we can flip up to half of the spins in parallel utilizing the
* many threads on the GPU.
*/
void Ising::monteCarlo3d()
{
    initializeSystem3d();

    while (iter < n_iters)
    {
        getRandom3d();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    calcDeltaHamiltonian3d(i,j,k);
                    if (dE <= 0 || R3d[i][j][k] <= exp(-1.0*dE))
                    {
                        A3d[i][j][k] *= -1;
                    }
                }
                
            }
        }

        iter++;
        calcHamiltonian3d();
        calcMagnetization3d();
    }

    // std::cout << "E = " << E[iter] << std::endl;
    // std::cout << "M = " << M[iter] << std::endl;

    // writeOutput();
    finalizeSystem3d();
}

void Ising::initializeSystem2d()
{
    N = n*n;

    A2d = new int *[n];
    R2d = new float *[n];

    for (int i = 0; i < n; i++)
    {
        A2d[i] = new int[n];
        R2d[i] = new float[n];
    }

    srand(time(NULL));
    /* Initialize spins on lattice */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A2d[i][j] = std::rand()%2 * 2 - 1;
        }
    }

    calcHamiltonian2d();
    calcMagnetization2d();
}

void Ising::initializeSystem3d()
{
    N = n*n*n;

    A3d = new int **[n];
    R3d = new float **[n];

    for (int i = 0; i < n; i++)
    {
        A3d[i] = new int*[n];
        R3d[i] = new float*[n];

        for (int j = 0; j < n; j++)
        {
            A3d[i][j] = new int[n];
            R3d[i][j] = new float[n];
        }
    }

    srand(time(NULL));
    /* Initialize spins on lattice */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                A3d[i][j][k] = std::rand()%2 * 2 - 1;
            }
        }
    }

    calcHamiltonian3d();
    calcMagnetization3d();
}

void Ising::finalizeSystem2d()
{

    for (int i = 0; i < n; i++)
    {
        delete[] A2d[i];
        delete[] R2d[i];
    }

    delete[] A2d;
    delete[] R2d;

}

void Ising::finalizeSystem3d()
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            delete[] A3d[i][j];
            delete[] R3d[i][j];
        }
        delete[] A3d[i];
        delete[] R3d[i];
    }
    delete[] A3d;
    delete[] R3d;
}
