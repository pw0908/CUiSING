#include "../Ising.h"

void Ising::getRandom2d()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            R2d[i][j] = (float) rand()/RAND_MAX;
        }
    }
}

void Ising::getRandom3d()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                R3d[i][j][k] = (float) rand()/RAND_MAX;
            }
        }
    }
}