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
    int d;
    double J;
    double h;
    int **A2d;
    int ***A3d;
    float **R2d;
    float ***R3d;

    /* Iteration Parameters */
    int n_iters;
    int sample_freq;
    int iter;

    /* Observables */
    std::vector<double> E;
    std::vector<double> M;
    double dE;

    /*=========================
     Data I/O
    =========================*/
    /* Folder names */
    std::string outputDir;
    
    /* File names */
    std::string outputFilename;

    /* Actual Files */
    std::string outputFile;

    /* std::ofstream to do the actual I/O */
    // std::ofstream &outputStream;

    void setOutputFileStructure();
    
public:
    /* Default constructor */
    Ising();

    /* Constructor */
    Ising(int n_iters_,
          int d_,
          int n_,
          double J_,
          double h_);

    /* destructor */
    ~Ising();

    /* Main engine */
    void initializeSystem2d();
    void monteCarlo2d();
    void finalizeSystem2d();

    void initializeSystem3d();
    void monteCarlo3d();
    void finalizeSystem3d();

    /* Calculations */
    void calcHamiltonian2d();
    void calcDeltaHamiltonian2d(int i, int j);
    void calcMagnetization2d();

    void calcHamiltonian3d();
    void calcDeltaHamiltonian3d(int i, int j, int k);
    void calcMagnetization3d();

    /* Helper */
    void getRandom2d();
    void getRandom3d();

    /* Data I/O */
    void openLog();
    void writeLog();
    void writeOutput();
    void finalizeIO();
};


#endif