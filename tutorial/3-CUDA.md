# Using CUDA in Julia
Using the CUDA.jl package, one is able to exploit GPU parallelisation within Julia. Thankfully, most CUDA based packages are available within CUDA.jl (CUBLAS, CURAND, CUFFT, _etc._). The CUDA.jl [documentation](https://cuda.juliagpu.org/stable/) provides an excellent introduction to the package however, as a very basic introduction, we will point out some of the key features here (specifically those used in `Ising.jl`). As a general rule, if a function exists in CUDA, it can be called exactly the same way in CUDA.jl.

One of the important things to point out is, to make CUDA.jl closer to how it is presented in C++, it is best to define functions using `!` at the end of the name, making it behave more like a compiled language. One requirement for CUDA.jl is that these functions must always end with `return`, otherwise an error will be thrown.

_N.B._ It is very important to remember that Julia is index-1 as this can have significant repercusion on defining indices. For example, obtaining the array index will now become:
```julia
tid = blockDim().x*(blockIdx().x-1) + threadIdx().x
```

## CuArrays
As with all things julia, in an attempt to abstract some of the complexities away, the developers of CUDA.jl have defined a special type of array called `CuArrays`. These can be defined as follows:
```julia
A = CuArray{Type}(undef,dims)
```
This procedure serves the same purpose as `cudaMalloc` in typical CUDA, only the memory size allocation and loading and off-loading from the host has been abstracted away. This allows the `CuArray` to be almost treated exactly like a regular array. Nevertheless, it is still possible to use `cudaMalloc` through `CUDA.Mem.alloc()` (with the same inputs).

## Running on CUDA
While in C++ we might write:
```cuda
cudaCalcMagnetization2dKernel<<<blocks,THREADS>>>(lattice,M,n,J,h,iter);
```
In CUDA.jl, this is written as:
```julia
@cuda blocks=n_blocks threads=n_threads calcMagnetisationKernel!(model,lattice,M,iter)
```
At this point, we introduce the concept of a macro (functions preceeded by `@`). These are quite complex pieces of code but, in short, these are functions which write code. More details on macros can be found [here](https://docs.julialang.org/en/v1/manual/metaprogramming/).

Nevertheless, we can see from the above that the structure between the two codes does not change too drastically.

## Shared memory
In CUDA.jl, one cannot simply add:
```cuda
extern __shared__ float shmem[];
```
To produce a shared memory. The process to creating an `shmem`-like object involves first changing the inputs to our `@cuda` macro:
```julia
@cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcHamiltonianKernel!(model,lattice,E,iter)
```
As we can see, we must define the size of our shared memory beforehand. Once this has been defined, within our kernel, we can add:
```julia
s = CuDynamicSharedArray(Float64,n_threads)
```
where `s` is now our shared memory array. If we want to ensure that our threads are synced before proceeding in the code, we can easily do so using a simple function call:
```julia
sync_threads()
```
Finally, atomic-level computations can be done easily using the following macro:
```julia
CUDA.@atomic A += b
```
where any of the standard arithmetic operators can be used.

## Tips and tricks
From our experience coding using CUDA.jl, we have the following comments to make:
* Within any kernel functions, if a line of code involves indexing an array, preface the line with the macro `@inbounds`. Whenever a particular matrix index is called, julia will first check that this index is within bounds. This can slow down the computational time and, as such, to avoid this, one can use this macro to tell julia not to perform this check.
* For the purposes of debugging, it is recommended to start the julia REPL using the following:
```
julia -g2
```
This will give much better error messages than the usual julia output but will be quite verbose.
