#include "global_external.h"
#include "global_internal.h"

using namespace SysInfo;
using namespace std;

int main(int argc, char *argv[])
{
    /* Initialize ising object */
    Ising ising;

    // Parse input arguments
    int n;
    double J;
    double h;

    if (argc > 3)
    {
        n = atof(argv[1]);
        J = atof(argv[2]);
        h = atof(argv[3]);
        cout << "===============" << endl;
        cout << "n:  " << n       << endl;
        cout << "J:  " << J       << endl;
        cout << "h:  " << h       << endl;
        cout << "===============" << endl;
        ising = Ising(n,J,h);
    }
    else if (argc > 2)
    {
        n = atof(argv[1]);
        J = atof(argv[2]);
        cout << "==============="   << endl;
        cout << "n:  " << n         << endl;
        cout << "J:  " << J         << endl;
        cout << "h:  " << "default" << endl;
        cout << "==============="   << endl;

    }
    else if (argc > 1)
    {
        n = atof(argv[1]);
        cout << "==============="   << endl;
        cout << "n:  " << n         << endl;
        cout << "J:  " << "default" << endl;
        cout << "h:  " << "default" << endl;
        cout << "==============="   << endl;
    }
    else
    {
        cout << "==================================" << endl;
        cout << "Using default system parameters"    << endl;
        cout << "==================================" << endl;
    }

    // Initialize timers
    Timers MainTimer;
    


    return 0;
}
