#include "../Ising.h"

void Ising::calcHamiltonian()
{
    double e = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            e -= J*A[i][j]*(A[i][(n+(j+1)%n)%n]+A[i][(n+(j-1)%n)%n]+
                            A[(n+(i+1)%n)%n][j]+A[(n+(i-1)%n)%n][j])/2.0+
                            A[i][j]*h;
        }
    }
    E[iter] = e;
}

void Ising::calcDeltaHamiltonian(int i, int j)
{
    dE = 2.0*(J*A[i][j]*(A[i][(n+((j+1)%n))%n]+A[i][(n+((j-1)%n))%n]+
                            A[(n+((i+1)%n))%n][j]+A[(n+((i-1)%n))%n][j])+
                            A[i][j]*h);
}

void Ising::calcMagnetization()
{
    int m = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m += A[i][j];
        }
    }
    M[iter] = m / (double)N;
}