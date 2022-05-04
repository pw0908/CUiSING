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
    int **points;

    /* Iteration Parameters */
    int n_iters;
    int sample_freq;
    int iter;

    /* Observables */
    double *E;
    double *m;


public:
    /* Default constructor/destructor*/
    Ising();
    ~Ising();

    /* Constructor */
    Ising(int n_ = 50,
          double J_ = 1.0,
          double h_ = 0.0,
          int n_iters_ = 1000,
          int sample_freq_ = 10);

    /* Main engine */
    void initializeSystem();
    void monteCarlo();
    void finalizeSystem();

    /* Calculations */
    void calcHamiltonian();
    void calcMagnetization();

    /* Helper */
    void getPoints();

    /* Data I/O */
    void openLog();
    void writeLog();
    void writeData();
    void finalizeIO();
};


#endif