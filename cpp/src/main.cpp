#include "global_external.h"
#include "global_internal.h"

using namespace SysInfo;
using namespace std;

int main(int argc, char *argv[])
{
    // Parse input arguments
    int n_iters = 1000;
    int n = 50;
    double J = 1.0;
    double h = 0.0;
    
    if (argc > 4)
    {
        n_iters = atof(argv[1]);
        n = atof(argv[2]);
        J = atof(argv[3]);
        h = atof(argv[4]);
        Ising ising = Ising(n_iters,n,J,h);
    }
    else if (argc > 3)
    {
        n_iters = atof(argv[1]);
        n = atof(argv[2]);
        J = atof(argv[3]);
        Ising ising = Ising(n_iters,n,J);

    }
    else if (argc > 2)
    {
        n_iters = atof(argv[1]);
        n = atof(argv[2]);
        Ising ising = Ising(n_iters,n);
    }
    else if (argc > 1)
    {
        n_iters = atof(argv[1]);
        Ising ising = Ising(n_iters);
    }

    cout << "=========================" << endl;
    cout << "n_iters:   " << n_iters    << endl;
    cout << "n:         " << n          << endl;
    cout << "J:         " << J          << endl;
    cout << "h:         " << h          << endl;
    cout << "=========================" << endl;

    // Initialize timers
    Timers MainTimer;
    
    return 0;
}
