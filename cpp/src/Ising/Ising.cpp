#include "Ising.h"

/* Default constructor */
Ising::Ising() = default;

/* Default destructor */
Ising::~Ising()
{
    delete[] E;
    delete[] m;
}

/* Constructor */
Ising::Ising(int n_ = 50,
             double J_ = 1.0,
             double h_ = 0.0,
             int n_iters_ = 1000,
             int sample_freq_ = 10)
{
    n = n_;
    J = J_;
    h = h_;
    n_iters = n_iters_;
    sample_freq = sample_freq_;

    E = new double[n_iters];
    m = new double[n_iters];

    iter = 0;
}


void Ising::monteCarlo()
{
    initializeSystem();
}

void Ising::initializeSystem()
{
    N = n*n;

    A = new int *[n];
    points = new int *[N];

    for (int i = 0; i < n; i++)
    {
        A[i] = new int[n];
        points[i] = new int[3];
    }

    for (int i = 0; i < N; i++)
    {
        points[i] = new int[3];
    }

    int spin;
    int count = 0;
    /* Initialize spins on lattice */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            spin = std::rand()%2 * 2 - 1;
            A[i][j] = spin;
            points[count][0] = i;
            points[count][1] = j;
            points[count][2] = spin;
            count++;
        }
    }

    calcHamiltonian();
    calcMagnetization();

}
