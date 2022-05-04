#include "../Ising.h"

void Ising::calcHamiltonian()
{
    e = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            
        }
    }
    
}

void Ising::calcMagnetization()
{
    m = 0.0;
    for (int i = 0; i < N; i++)
    {
        m += points[i][3];
    }
    m /= (double)N;
}