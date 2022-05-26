# CUiSING
Welcome to CUiSING! A CUDA-parallelised implementation of the Ising model written in Python, Julia and C++.

This package was developed as part of the CS 179 course requirement in Caltech.

The Ising model is a lattice model in which spins interact via magnetic interactions, and also interact with an externally applied external field. Spins can be in one of two states, up or down. The Hamiltonian for the simple Ising model is,
$$H(\{\mathbf{s}\})=-\frac{J}{2}\sum_i\sum_j s_i s_j - h\sum_i s_i$$
where the first sum is over all spins, and their nearest neighbors.

In Markov Chain Monte Carlo (MCMC) a series of moves are attempted, and accepted based on probabilistic criteria to drive the system to the lowest free energy state. In this package, we have implemented the Metropolis-Hastings algorithm which uses the Boltzmann weight from statistical mechanics combined with detailed balance.
## CPU Demo

This package features CPU based implementations in three different programming languages; C++, Julia, and Python. The purpose is to compare the CPU performance of basic Ising monte carlo simulations. Later, parallelization and GPU performance will be compared between the 3 languages as well.

Each of the cpu codes are self contained in their respective folders within CUiSING/. The following sections describe how to run each of the CPU codes.

### C++

To run the C++ code, you must first enter the cpp directory, and compile the C++ code (if not already compiled)
```
cd cpp
make

```
The C++ executable can be run with the following command and the following flags. It can also be called without any flags and the default parameters will be used.
```
./Ising <n_iters> <d> <n> <J> <h>
```
- n_iters = number of MC iterations (default: 1000)
- d = spacial dimension, 2 or 3 (default: 2)
- n = number of spins along single direction (default: 50)
- J = magnetic interaction strength (default: 1.0)
- h = external field strength (default: 0.0)

The energy and magnetization trajectories are stored in ```cpp/output/output.dat``` and plotted in ```cpp/figures/Sys_<n_iter>_<d>_<n>_<J>_<h>/```.

### Julia
To run the Julia code, first enter the julia folder
```
cd Julia
```
From within the Julia folder you can run the Julia code with the following command. You must include values for all of the flags.
```
julia benchmarking.jl <n_iters> <d> <n> <J> <h>
```
### Python
To run the python code you can first enter the python directory,
```
cd python
```
the code can then be run with the following command. You must include values for all of the flag.
```
python benchmarking.py <n_iters> <d> <n> <J> <h>
```

### CPU Benchmarking
Thusfar we have done some work with benchmarking the CPU implementation across different hardware and across the three different languages. The data and figures are stored in ```benchmarking/``` and they are stored by dimension.

The 2d and 3d results as run on Sam's PC (Intel i9-10850k) are given below,
![Sam 2d benchmarks](benchmarking/2d/figures/Sam_1000_0.1_0_2.png)
![Sam 3d benchmarks](benchmarking/3d/figures/Sam_1000_0.1_0_3.png)
Note that we do recover the proper $n^d$ scaling as expected without any parallelization. This is because each MC iteration requires a loop over all $n^d$ particles in the system.

Pierre has also run some benchmarks with similar results both on his PC (Ryzen 5900X) and his 2022 Macbook Pro (ARM).

Benchmarks of all of the methods can be run simultaneously by using the ```CUiSING/benchmarking.py``` within the ```CUiSING/``` folder. You can change the system and iterative parameters within the python file, and then simply run the following command.
```
python benchmarking.py
```
The span of $n$ to be benchmarked can be set using the logspace command in line 48 of the script.
```
n = np.logspace(low,high,number).astype(int)
```

## GPU Implementation
Coming soon...

The C++ code has comments above the functions which will be parallelized on the GPU. We plan to implement the following parallelizations in all languages:
- Parallelizing the Hamiltonian calculation by parallelizing the calculation of individual spin energies, using a reduction to add them per block, and then using an atomic add to add up the contributions from each block.
- Parallelizing the Monte Carlo (MC) loop using a checkerboard stencil in which many MC moves can be attempted at once due to the small range of the interactions. Spins that are outside the nearest neighbor cutoff can be flipped simultaneously since their states are independent. This allows us to attempt flipping up to half of the spins at the same time. This will drastically speed up the implementation.
- Lastly, we generate large arrays of random numbers, which can be parallelized on the GPU to generate all of the random numbers at the same time. There are algorithms which can generate arrays in parallel without relying on the system time.
  
We haven't included comments about parallelization in Julia and python because the structure is the same, and the parallelization will be done in the same way across the languages. The difference willl be in the CUDA interface used, whether it be CUDA (C++), pyCUDA (python), or CUDA.jl (Julia). We will implement the same parallelizations across the three different languages using the available functions, and compare their performance. We suspect that the C++-CUDA code will perform the fastest, however, if the Julia-CUDA.jl implementation is close to the performance of the C++-CUDA code, then a case can be made for using Julia since both the CPU and GPU syntax is extremely simple and easy to pick up.