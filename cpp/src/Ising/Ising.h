#ifndef ISING_H
#define ISING_H

/* Include utilities */
#include "../global_external.h"
#include "../utilities/utilities.h"

/* Class that holds variables and functions for Ising MC */
class Ising
{
private:
    /*=========================
    Ising Variables
    =========================*/
    /* System variables */
    int n;
    int N;
    double J;
    double h;
    int dim = 2;
    int **A;
    float **R;

    /* Iteration Parameters */
    int n_iters;
    int sample_freq;
    int iter;

    /* Observables */
    std::vector<double> E;
    std::vector<double> M;
    double dE;


public:
    /* Default constructor */
    Ising();

    /* Constructor */
    Ising(int n_iters_,
          int n_,
          double J_,
          double h_);

    /* destructor */
    ~Ising();

    /* Main engine */
    void initializeSystem();
    void monteCarlo();
    void finalizeSystem();

    /* Calculations */
    void calcHamiltonian();
    void calcDeltaHamiltonian(int i, int j);
    void calcMagnetization();

    /* Helper */
    void getPoints();
    void getRandom();

    /* Data I/O */
    void openLog();
    void writeLog();
    void writeData();
    void finalizeIO();
};


#endif