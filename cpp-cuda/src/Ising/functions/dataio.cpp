#include "../Ising.h"

void Ising::setOutputFileStructure()
{
    outputDir = "output/";
    outputFilename = "output.dat";

    struct stat sb;
    if (stat(outputDir.c_str(), &sb) == -1)
    {
        std::string mkdirBase = "mkdir " + outputDir;
        int systemRet = std::system(mkdirBase.c_str());
        if (systemRet == -1)
        {
            std::cerr << "Ising : failed to create output directory" << std::endl;
            exit(0);
        }
    }
}

void Ising::writeOutput()
{
    std::ofstream outputStream;
    std::stringstream ss;
    outputFile = outputDir+outputFilename;
    outputStream.open(outputFile.c_str(), std::fstream::out | std::ofstream::trunc);

    for (int j = 0; j < n_iters+1; j+=sample_freq)
    {
        outputStream << j << " ";
        outputStream << std::fixed << std::setprecision(5) << E[j] << " ";
        outputStream << std::fixed << std::setprecision(5) << M[j] << " \n";
    }

    // Close the file
    outputStream.close();
}
