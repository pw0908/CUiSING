#include "../Ising.h"

/*
* Calculating these random numbers can be done
* in parallel using a variety of algorithms. So long
* as they are not based on system time.
*/
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

/*
* Calculating these random numbers can be done
* in parallel using a variety of algorithms. So long
* as they are not based on system time.
*/
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