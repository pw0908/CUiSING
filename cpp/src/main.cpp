#include "global_external.h"
#include "global_internal.h"

using namespace SysInfo;
using namespace std;

int main(int argc, char *argv[])
{
    // Parse input arguments
    int n_iters = 1000;
    int d = 2;
    int n = 50;
    double J = 1.0;
    double h = 0.0;
    Ising ising;
    
    if (argc > 5)
    {
        n_iters = atof(argv[1]);
        d = atof(argv[2]);
        n = atof(argv[3]);
        J = atof(argv[4]);
        h = atof(argv[5]);
        ising = Ising(n_iters,d,n,J,h);
    }
    else if (argc > 1)
    {
        std::cerr << "Need to supply 5 arguments (or none): int n_iters, int n, double J, double h" << std::endl;
        std::exit(0);
    }
    else
    {
        ising = Ising();
    }

    cout << "=========================" << endl;
    cout << "n_iters:   " << n_iters    << endl;
    cout << "d:         " << d          << endl;
    cout << "n:         " << n          << endl;
    cout << "J:         " << J          << endl;
    cout << "h:         " << h          << endl;
    cout << "=========================" << endl;

    // Initialize timers
    Timers MainTimer;

    START_CPU_TIMER(MainTimer);

    if (d == 2)
    {
        ising.monteCarlo2d();
    }
    else if (d == 3)
    {
        ising.monteCarlo3d();
    }
    else
    {
        std::cerr << "Dimension is something other than 2 or 3...  d = " << d << std::endl;
        std::exit(0);
    }

    STOP_CPU_TIMER(MainTimer);

    ising.writeOutput();

    int number1 = MainTimer.duration;
    std::stringstream ss1;
    ss1 << number1;

    std::cout << "\nTotal Program Time : " << MainTimer.duration << " seconds" << std::endl;
    
    return 0;
}
