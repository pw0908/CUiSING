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


