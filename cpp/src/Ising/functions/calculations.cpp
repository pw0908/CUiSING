#include "../Ising.h"

void Ising::calcHamiltonian2d()
{
    double e = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            e -= J*A2d[i][j]*(A2d[i][(n+(j+1)%n)%n]+A2d[i][(n+(j-1)%n)%n]+
                            A2d[(n+(i+1)%n)%n][j]+A2d[(n+(i-1)%n)%n][j])/2.0+
                            A2d[i][j]*h;
        }
    }
    E[iter] = e;
}

void Ising::calcHamiltonian3d()
{
    double e = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                e -= J*A3d[i][j][k]*(A3d[i][(n+(j+1)%n)%n][k]+A3d[i][(n+(j-1)%n)%n][k]+
                                     A3d[(n+(i+1)%n)%n][j][k]+A3d[(n+(i-1)%n)%n][j][k]+
                                     A3d[i][j][(n+(k+1)%n)%n]+A3d[i][j][(n+(k-1)%n)%n])/2.0+
                                     A3d[i][j][k]*h;

            }
        }
    }
    E[iter] = e;
}

void Ising::calcDeltaHamiltonian2d(int i, int j)
{
    dE = 2.0*(J*A2d[i][j]*(A2d[i][(n+((j+1)%n))%n]+A2d[i][(n+((j-1)%n))%n]+
                            A2d[(n+((i+1)%n))%n][j]+A2d[(n+((i-1)%n))%n][j])+
                            A2d[i][j]*h);
}

void Ising::calcDeltaHamiltonian3d(int i, int j, int k)
{
    dE = 2.0*(J*A3d[i][j][k]*(A3d[i][(n+(j+1)%n)%n][k]+A3d[i][(n+(j-1)%n)%n][k]+
                              A3d[(n+(i+1)%n)%n][j][k]+A3d[(n+(i-1)%n)%n][j][k]+
                              A3d[i][j][(n+(k+1)%n)%n]+A3d[i][j][(n+(k-1)%n)%n])+
                              A3d[i][j][k]*h);
}

void Ising::calcMagnetization2d()
{
    int m = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m += A2d[i][j];
        }
    }
    M[iter] = m / (double)N;
}

void Ising::calcMagnetization3d()
{
    int m = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                m += A3d[i][j][k];
            }
        }
    }
    M[iter] = m / (double)N;
}