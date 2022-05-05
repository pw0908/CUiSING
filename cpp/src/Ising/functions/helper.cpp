#include "../Ising.h"

void Ising::getRandom()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            R[i][j] = (float) rand()/RAND_MAX;
        }
    }
}