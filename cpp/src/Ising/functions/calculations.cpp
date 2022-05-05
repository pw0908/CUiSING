#include "../Ising.h"

void Ising::calcHamiltonian()
{
    E[iter] = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            E[iter] -= J*A[i][j]*(A[i][(n+((j+1)%n))%n]+A[i][(n+((j-1)%n))%n]+
                            A[(n+((i+1)%n))%n][j]+A[(n+((i-1)%n))%n][j])/2.0+
                            A[i][j]*h;
        }
    }
}

void Ising::calcDeltaHamiltonian(int i, int j)
{
    dE = 2*(J*A[i][j]*(A[i][(n+((j+1)%n))%n]+A[i][(n+((j-1)%n))%n]+
                            A[(n+((i+1)%n))%n][j]+A[(n+((i-1)%n))%n][j])/2.0+
                            A[i][j]*h);
}

void Ising::calcMagnetization()
{
    M[iter] = 0.0;
    for (int i = 0; i < N; i++)
    {
        M[iter] += points[i][3];
    }
    M[iter] /= (double)N;
}