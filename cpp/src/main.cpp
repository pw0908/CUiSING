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
    Ising ising;
    
    if (argc > 4)
    {
        n_iters = atof(argv[1]);
        n = atof(argv[2]);
        J = atof(argv[3]);
        h = atof(argv[4]);
        ising = Ising(n_iters,n,J,h);
    }
    else if (argc > 1)
    {
        std::cerr << "Need to supply 4 arguments (or none): int n_iters, int n, double J, double h" << std::endl;
        std::exit(0);
    }
    else
    {
        ising = Ising();
    }

    cout << "=========================" << endl;
    cout << "n_iters:   " << n_iters    << endl;
    cout << "n:         " << n          << endl;
    cout << "J:         " << J          << endl;
    cout << "h:         " << h          << endl;
    cout << "=========================" << endl;

    // Initialize timers
    Timers MainTimer;

    START_CPU_TIMER(MainTimer);

    ising.monteCarlo();

    STOP_CPU_TIMER(MainTimer);

    int number1 = MainTimer.duration;
    std::stringstream ss1;
    ss1 << number1;

    std::cout << "\nTotal Program Time : " << MainTimer.duration << " seconds" << std::endl;
    
    return 0;
}
