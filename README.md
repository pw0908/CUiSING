# CUiSING
Welcome to CUiSING! A CUDA-parallelised implementation of the Ising model written in Python, Julia and C++.

This package was developed as part of the CS 179 course requirement in Caltech.

The Ising model is a lattice model in which spins interact via magnetic interactions, and also interact with an externally applied magnetic field. Spins can be in one of two states, up (+) or down (-). The Hamiltonian for the simple Ising model is,
$$H(\{\mathbf{s}\})=-\frac{J}{2}\sum_i\sum_j s_i s_j - h\sum_i s_i$$
where the first sum is over all spins, and their nearest neighbors.

In Markov Chain Monte Carlo (MCMC) a series of moves are attempted, and accepted based on probabilistic criteria to drive the system to the lowest free energy state. In this package, we have implemented the Metropolis-Hastings algorithm which uses the Boltzmann weight from statistical mechanics combined with detailed balance. During the simulation, the spin state will evolve with each iteration, and, therefore, so will the total energy and magnetization. Detailed balance ensures that the system evolves to the equilibrium state. The energy and magnetization are the main observables of the system, and they are defined as follows,

$$M=\frac{1}{N}\sum_i s_i$$
$$E=H=-\frac{J}{2}\sum_i\sum_j s_i s_j - h\sum_i s_i$$

We normalize the energy such that in a fully aligned system the normalized energy is -1. This is useful for testing purposes so that all trajectories that undergo a transition from random to aligned should have a normalized equilibrium energy of -1.

$$\bar E =\frac{E}{JNz/2+|h|}=\frac{E}{JNd+|h|}$$

where $z$ is the coordination number of the lattice (4 in 2d and 6 in 3d). In 2d and 3d, we have $d=z/2$.

## Overview

This package contains many different implementations in different languages that run the same simulation. The goal of each one is the same, to perform Metropolis Monte Carlo on the 2d and 3d Ising model. The purpose of the different implementations is to show differences in speed among various languages on CPU and GPU. Here, we list all the available codes with a brief description of each:
- **cpp**: a C++ implementation, run fully on CPU
- **cpp-cuda**: a CUDA implementation parallelized on GPU. The GPU kernels are wrapped in C++ code and functions.
  - Requires  cuRAND package, which comes loaded with CUDA
- **julia**: a Julia implementation, run fully on CPU
- **julia-cuda**: a CUDA implementation parallelized on GPU. The GPU kernels are wrapped in Julia code and functions. This makes use of **CUDA.jl**
- **python**: a Python implementation, run fully on CPU
- **matlab**: a vectorized matlab implementation, run fully on CPU
- **benchmarking**: contains benchmarking data for CPU and GPU code.
- **tutorial**: Tutorial for setting up and using Julia, with special emphasis on how to run **CUDA.jl**.
- **demos_for_TAs**: folder that contains scripts for running a simple 2d simulation on the GPU in C++ and Julia
- **production_run**: code for a production level set of simulations to extract the $M$ vs $J$ curve. This was written as a test of how the code fairs in a realistic scientific situation.

## Installation Instructions

Before getting started, there are some brief installation procedures that need to be followed for the different languages:

#### C++ (Sam)
The C++ code should be runable with C++ versions $\ge$ 11. No additional libraries need to be downloaded.

#### Julia (Pierre)
First you must install Julia onto the machine. There is a guide on how to do this for Linux, MacOS, and Windows available in ```tutorial/```. Once Julia is installed, CUDA.jl needs to be added in order to run the GPU code. Install the CUDA package by running the following from the command line,
```
julia
] add CUDA LinearAlgebra
```
You should now be all set to run the Julia programs.

#### Python (Sam +Pierre)
The python code makes use of ```numpy```. Install it by running the following commands:
```
pip install numpy
```

#### Benchmarking/Wrappers
Many of the wrappers and benchmarking scripts written in Python make use of additional packages. Install them with the following commands,
```
pip install matplotlib progressbar csv
```

## CPU Demo

This package features CPU based implementations in three different programming languages; C++, Julia, and Python. The purpose is to compare the CPU performance of basic Ising monte carlo simulations. Later, parallelization and GPU performance will be compared between the 3 languages as well.

Each of the cpu codes are self contained in their respective folders within CUiSING/. The following sections describe how to run each of the CPU codes.

### C++ (Sam)

To run the C++ code, you must first compile it (if not already compiled). Assuming you are in ```CUiSING/```, the following command will build the C++ executable ```Ising``` within the ```cpp/``` directory.
```
make -C cpp
```
The C++ executable can be run with the following command and the following flags. It can also be called without any flags and the default parameters will be used.
```
./cpp/Ising <n_iters> <d> <n> <J> <h>
```
- **n_iters** = number of MC iterations (default: 1000)
- **d** = spacial dimension, 2 or 3 (default: 2)
- **n** = number of spins along single direction (default: 50)
- **J** = magnetic interaction strength (default: 1.0)
- **h** = external field strength (default: 0.0)

The energy and magnetization trajectories are stored in ```output/cpp_cpu_output.dat```.

### Julia (Pierre)
To run the Julia code from the ```CUiSING/``` directory, open the Julia REPL and use the following code:
```julia
include("julia/Ising.jl")
```
One can then define the Ising  `model`, depending on the dimension, as:
```julia
model = Ising2DParam(n, J, h, n_iters)
```
To simulate the system, for both 2D and 3D, simply run the following:
```julia
ms, Es = MCIsing(model)
```
Where `Es` and `ms` are the energy and magnetization, respectively.

From the command line, one can also directly call:
```
julia julia/benchmarking.jl <n_iters> <d> <n> <J> <h>
```
The energy and magnetization trajectories are stored in ```output/julia_cpu_output.dat```.

### Python (Sam + Pierre)
To run the Python code from the ```CUiSING/``` directory, run the following command (you must include values for all of the arguments),
```
python3 python/benchmarking.py <n_iters> <d> <n> <J> <h>
```
The energy and magnetization trajectories are stored in ```output/python_cpu_output.dat```. Note that the python code is very slow compared to the C++ and Julia code, so you might need to decrease the system size in order to get an output in a reasonable time.

### CPU Benchmarking (Sam + Pierre)
Thusfar we have done some work with benchmarking the CPU implementation across different hardware and across the three different languages. The data and figures are stored in ```benchmarking/``` and they are stored by dimension.

The 2d and 3d results as run on Sam's PC (Intel i9-10850k) are given below,

![Sam 2d benchmarks](benchmarking/2d/figures/Sam_CPU_1000_0.1_0_2.png)

![Sam 3d benchmarks](benchmarking/3d/figures/Sam_CPU_1000_0.1_0_3.png)

Note that we do recover the proper $n^d$ scaling as expected without any parallelization. This is because each MC iteration requires a loop over all $n^d$ particles in the system.

Pierre has also run some benchmarks with similar results both on his PC (Ryzen 5950X):
![Pierre 2d benchmarks](benchmarking/2d/figures/Pierre_1000_0.1_0_2.png)
![Pierre 3d benchmarks](benchmarking/3d/figures/Pierre_1000_0.1_0_3.png)
 and his 2022 Macbook Pro (ARM):
![Pierre ARM 2d benchmarks](benchmarking/2d/figures/Pierre_ARM_1000_0.1_0_2.png)
![Pierre ARM 3d benchmarks](benchmarking/3d/figures/Pierre_ARM_1000_0.1_0_3.png)
Benchmarks of all of the methods can be run simultaneously by using the ```CUiSING/benchmarking.py``` within the ```CUiSING/``` folder. You can change the system and iterative parameters within the python file, and then simply run the following command.
```
python3 benchmarking.py
```
The span of $n$ to be benchmarked can be set using the logspace command in line 48 of the script.
```
n = np.logspace(low,high,number).astype(int)
```

## GPU Implementation Intro/Goals

*This part was submitted with the cpu demo at the midpoint of the project. We have since implemented these things. Read beyond for information and results!*

The C++ code has comments above the functions which will be parallelized on the GPU. We plan to implement the following parallelizations in all languages:
- Parallelizing the Hamiltonian calculation by parallelizing the calculation of individual spin energies, using a reduction to add them per block, and then using an atomic add to add up the contributions from each block.
- Parallelizing the Monte Carlo (MC) loop using a checkerboard stencil in which many MC moves can be attempted at once due to the small range of the interactions. Spins that are outside the nearest neighbor cutoff can be flipped simultaneously since their states are independent. This allows us to attempt flipping up to half of the spins at the same time. This will drastically speed up the implementation.
- Lastly, we generate large arrays of random numbers, which can be parallelized on the GPU to generate all of the random numbers at the same time. There are algorithms which can generate arrays in parallel without relying on the system time.
  
We haven't included comments about parallelization in Julia  because the structure is the same, and the parallelization will be done in the same way. The difference willl be in the CUDA interface used, whether it be CUDA (C++) or CUDA.jl (Julia). We suspect that the C++-CUDA code will perform the fastest, however, if the Julia-CUDA.jl implementation is close to the performance of the C++-CUDA code, then a case can be made for using Julia since both the CPU and GPU syntax is extremely simple and easy to pick up.

## GPU Results

### C++ with CUDA (Sam)
The C++ CUDA implementation is located the ```cpp-cuda/``` directory. Within the ```cpu-cuda/src``` directory, there are a number of files:
- **main.cu**: the main script which reads in user input and execuates the simulation by calling functions
- **ising.h**: header file to include basic libraries, as well as libraries specific to this project, such as **cuRAND**.
- **cuda_helper.h**: includes helper functions for checking and interpreting CUDA and cuRAND errors
- **ising2d.h** and **ising3d.h**: contains declarations of host wrapper functions and cuda kernels for 2d and 3d ising model simulations, respectively.
- **ising2d.cu** and **ising3d.cu**: contains function and kernel definitions for 2d and 3d ising model simulations, respectively.

Before running any simulations, the code must be compiled. Depending on the machine, you may need to make changes to the **makefile**, particularly the **CUDA lib and include paths**, the **nvcc path** and the **CUDA gencodes**. The proper gencodes for each GPU architecture and CUDA version can be found at the following link: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/.

To compile the ```cpp-cuda``` code, run the following command from the ```CUiSING/``` directory,
```
make -C cpp-cuda
```

To run a simulation, run the following command
```
./cpp-cuda/Ising <n_iters> <d> <n> <J> <h>
```

- **n_iters** = number of MC iterations (default: 1000)
- **d** = spacial dimension, 2 or 3 (default: 2)
- **n** = number of spins along single direction (default: 100)
- **J** = magnetic interaction strength (default: 1.0)
- **h** = external field strength (default: 0.0)

If no arguments are provided then the default values will be used. The energy and magnetization trajectories are printed to the file ```output/cpp_gpu_output.dat```. These show how the system state evolved over "time".

### Julia with CUDA.jl (Pierre)
The Julia CUDA.jl implementation can be found under `julia-cuda/Ising.jl`. To avoid repetition, code relies in part on other functions defined in `julia/Ising.jl` for the CPU. 

Assuming all the steps mentioned previously (installing CUDA and CUDA.jl), the code should work as given. In order to use this code, from `CUiSING/`, open the Julia REPL and use the following:
```julia
include("julia-cuda/Ising.jl")
```
As it was for the CPU code, one first needs to construct the Ising `model`:
```julia
model = CUDAIsing2DParam(n, J, h, n_iters, n_threads)
```
Now we also include the optional argument `n_threads`. To simulate the system, for both 2D and 3D, simply run the following:
```julia
ms, Es = MCIsing(model)
```
Where `Es` and `ms` are the energy and magnetization, respectively.

From the command line, one can also directly call:
```
julia julia-cuda/benchmarking.jl <n_iters> <d> <n> <J> <h>
```
The energy and magnetization trajectories are stored in ```output/julia_gpu_output.dat```.
#### C++ with CUDA Benchmarks (Sam)

Below we compare the speeds of the ```cpp``` and ```cpp-cuda``` code over a large range of system sizes, in both 2d and 3d. The specifications for the CPU and GPU are:
- CPU: Intel i9-10850k (4.8 GHz boost clock, 10 cores, 20 threads)
- GPU: NVIDIA RTX 3080Ti
  - CUDA Version 11.4

![cpp vs cpp-cuda 2d](benchmarking/2d/figures/cpp_cpp-cuda_comparison.png)
![cpp vs cpp-cuda 3d](benchmarking/3d/figures/cpp_cpp-cuda_comparison.png)

We can see the incredible performance of the GPU implementation, especially for higher values of $n$. With the GPU code being 2 orders of magnitude faster than the C++ code, which was already the fastest CPU implementation. We do see at very small system sizes that the CPU code performs better, which is expected since there are not very many computations to parallelize. Another attractive feature of the GPU code is that the computation time remains fixed up to a certain system size. Since they are being run in parallel, adding more spins does not increase the computational time until the system becomes too big and then each thread has to handle multiple spins, and at that point, time begins to scale with the system size.

#### Example for ```cpp-cuda```: Spontaneous Magnetization! (Sam)

In 2d and 3d, the Ising model is well known for its interesting phase behavior. It turns out, that at high enough interaction strengths, $J$ (and/or low enough temperatures), the system will undergo a spontaneous transition from completely disorded to ordered without applying any external driving force ($h=0$). In the disordered state, all of the spins are randomly oriented, and therefore in the equilibrium state, the system fluctuates around a net magnetization of 0. However, if $J$ is high enough, then the system will spontaneously order, and the magnetization will become non-zero, as more spins face one direction than the other. Analysis of the partition function using statistical mechanical methods allows us to predict what $J$ needs to be in order for this transition to happen, we call this the critical $J$, or $J_c$. In 2d, this value is:
$$
J_c=\frac{\ln{\left(1+\sqrt{2}\right)}}{2}\approx 0.44
$$
We attempt to recover this spontaneouos phase transitions as well as the analytical critical interaction strength through the use of our GPU based MC simulations.

We ran numerous simulations at different interaction strengths, and extracted the mean magnetization at equilibrium from each one via block averaging. Besides $J$, The system parameters for each simulation remained fixed:
- $n_{\text{iters}}$ = $1e6$
- $d$ = $2$
- $n$ = $200$
- $h$ = $0$

$J$ was varied between $0.01-1.0$ amongst 40 different simulations. Below we plot the **energy** and **magnetization** trajectories for the 40 simulations, and finally, the plot of $M$ vs $J$,

![Sam Production E vs i](production_run/figures/E_vs_i.png)
![Sam Production M vs i](production_run/figures/M_vs_i.png)
![Sam Production M vs J](production_run/figures/M_vs_J.png)

We indeed see exactly what we expected. The energy starts high and decreases until plateuing out at an equilibrium value. The magnetization starts out at 0 (the system is initialized in random configuration), and as the MC simulation progresses, the system eithers stays disordered ($m=0$), or the magnetization grows until reaching an equilibrium value.

We also recover the phase behavior of the 2d Ising Model. The magnetization spontaneously becomes non-zero at $J=J_c\approx 0.44$. This indicates to us the success of our ```cpp-cuda``` code. Not only does it give the correct results, but in a fraction of the time as the CPU would have. Acquiring this plot of $M$ vs $J$ requires many simulations of large systems, run for an extremely large number of iterations. In order to ensure that the system reaches its true equilibrium, we ran for $1e6$ iterations. These calculations would have been extremely time-consuming on the CPU, and likely would have taken many hours or even days,  rather than minutes. Thus, we can see the benefit and necessity of using the GPU for these simulations.

Lastly, we take a look at the lattice itself at different stages throughout the MC simulation, for a system that undergoes spontaneous magnetization.

Beginning:
![Sam Lattice Beginning](production_run/figures/cpp_gpu_lattice_initial.png)
Middle:
![Sam Lattice Beginning](production_run/figures/cpp_gpu_lattice_intermediate.png)
End:
![Sam Lattice Beginning](production_run/figures/cpp_gpu_lattice_final.png)

#### cpp-cuda TA Demo: Simple 2d Simulation on GPU (Sam)
I have prepared a simple script for running a 2d simulation on the GPU using the cpp-cuda code. To run the simulation, you should do the following,starting from ```CUiSING/```:
```
make -C cpp-cuda
cd demos_for_TAs/cpp-cuda
```
Change the parameters withing ```ising_mc_demo_2d.py``` as you wish. I restrict this demo to 2d so that the lattice can be easily plotted/visualized. Then run the simulation with the following:
```
python ising_mc_demo_2d.py
```
The trajectory and lattice data are stored in ```demos_for_TAs/output/```. Figures showing the trajectories and the final lattice are stored in ```demos_for_TAs/figures```. Examples are shown below,

![Sam TA Demo Traj](demos_for_TAs/cpp-cuda/figures/trajectories.png)
![Sam TA Demo Latt](demos_for_TAs/cpp-cuda/figures/lattice.png)

#### julia-cuda TA Demo: Simple 2d Simulation on GPU (Pierre)
Similarly, an example simulation in Julia using CUDA has been provided in `demos_for_TAs/julia_cuda/ising_mc_demo_2d.jl`. No compilation will be needed as this is all handled by Julia.

One can alter the parameters within `ising_mc_demo_2d.jl` to simulate different systems if so desired. To run it from the `CUiSING/` directory, simply open the Julia REPL and run the following:
```julia
include("demos_for_TAs/julia_cuda/ising_mc_demo_2d.jl")
```
The figures from the simulation will be generated in `demos_for_TAs/julia_cuda/figures`. Examples are shown below:
![Pierre TA Demo Traj](demos_for_TAs/julia_cuda/figures/trajectories.png)
![Pierre TA Demo Latt](demos_for_TAs/julia_cuda/figures/lattice.png)

## Comparison between CPU and GPU, C++ and Julia
When setting out on this project, we wanted to compare two aspects: 1. the CPU and GPU implementations of the Ising Monte Carlo Simulation and 2. The C++ and Julia implementations of these codes. This comparison can be summarised in the figure below:
![Pierre CPU GPU benchmark](benchmarking/2d/figures/Pierre_gpu_cpu_comparison.png)
The above figure was generated for the 2D system using Pierre's PC with the following configurations (256 threads were used):
* CPU: AMD Ryzen 3950X
* GPU: MSi GEFORCE RTX 3070 Ti

As we can see from the figure, for small system sizes, the GPU implementation can be slower than the CPU implementation. This is unsurprising as the system is currently too small to benefit from parallelisation. However, for these small system sizes, the computational time is effectively constant where, for sizes approaching approximately $n=20$, the GPU code becomes faster than the CPU. 

It isn't until we reach orders of magnitude comparable to our number of threads that we finally begin to recover the expected $n^2$ trend. In this region, the GPU code is 1-2 orders of magnitude faster than the CPU, highlighting the incredible speed up from the techniques described earlier. 

Comparing both the Julia and C++ implementation, it should come to no surprise that the C++ implementation is faster, both for the CPU and GPU. The difference between the two languages is less appreciable in the CPU compared to the GPU; this possibly relates to how CUDA.jl interfaces with the GPU, leading to larger overhead (which is a general issue with Julia). In the GPU, the difference between the two languages is approximately a factor of 2. However, it is worth considering that the Julia code only required approximately half the number of lines of code as the C++ equivalent and it is, as an interpreted language, arguably, easier to read. This highlights the balance between computational speed and conciseness and, based on the graphs above, it is a tough judgement call as to which language is 'better' to use.

In terms of improvements, there are some easy ones related to the formulation of the problem. For example, rather than calculating the magnetisation and energy every iteration, one could simply calculate the energy and magnetisation change and add this to the value of the previous iteration. This could reduce the computational cost substantially. However, for the GPU, the 'checkerboard' strategy may be more-difficult to implement as all the energy changes are computed at once. 

On the Julia side, many of CUDA.jl's high-level features were used, rather than the low-level functions which directly interface with CUDA. This could bring the Julia code much closer to C++ in terms of computational time, at the loss of Julia's nice, concise code.