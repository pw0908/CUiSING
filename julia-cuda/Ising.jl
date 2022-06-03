include("../julia/Ising.jl")

using CUDA
import CUDA; CURAND,CUBLAS

abstract type CUDAIsingModel <: IsingModel end
abstract type CUDAIsing2DModel <: CUDAIsingModel end
abstract type CUDAIsing3DModel <: CUDAIsingModel end

struct CUDAIsing2DParam <: CUDAIsing2DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    n_threads::Int64
    N::Int64
    function CUDAIsing2DParam(n::Int64,J::Float64,h::Float64,
                              n_iters::Int64,n_threads::Int64)
        return new(n,J,h,n_iters,n_threads,n^2)
    end
end

struct CUDAIsing3DParam <: CUDAIsing3DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    n_threads::Int64
    N::Int64
    function CUDAIsing3DParam(n::Int64,J::Float64,h::Float64,
        n_iters::Int64,n_threads::Int64)
        return new(n,J,h,n_iters,n_threads,n^3)
    end
end 
"""
    InitialiseIsing(model::CUDAIsingModel)

    Initialise the Ising model for a CUDA lattice.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model 
    Outputs:
        - lattice: CuArray representing the spins in the lattice
        - rng: Random number generator used in the Metropolis algorithms
        - ms: Net magnetisation at each iteration
        - Es: Total energy at each iteration
"""
function InitialiseIsing(model::CUDAIsingModel)
    N = model.N
    n_iters = model.n_iters
    n_threads = model.n_threads
    n_blocks = Int64(floor((N+n_threads-1)/n_threads))

    ms = CUDA.zeros(Float64,n_iters + 1) 
    Es = CUDA.zeros(Float64,n_iters + 1)

    rng = CURAND.RNG(CURAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
    Random.seed!(rng,Int64(floor(time())))

    rands = Random.rand(rng, Float64, N)

    lattice = CuArray{Int}(undef,N)
    @cuda threads=n_threads blocks=n_blocks init_lattice!(model,lattice,rands)
    device_synchronize()
    return lattice, rng, ms, Es
end
"""
    init_lattice!(model::CUDAIsingModel,lattice,rands)

    The kernel for initialising the Ising model in CUDA.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model 
        - lattice: CuArray representing the spins in the lattice
        - rands: The previously-generated random numbers which 
          determine the spin in the lattice
"""
function init_lattice!(model::CUDAIsingModel,lattice,rands)
    N = model.N
    tid = blockDim().x*(blockIdx().x-1) + threadIdx().x
    if tid >= floor(N+1)
        return 
    end
    r = rands[tid];
    if r<0.5
        lattice[tid]=-1
    else
        lattice[tid]=1
    end
    return
end
"""
    calcHamiltonian!(model::CUDAIsingModel,lattice,E,iter)

    Calculates the total Energy for a lattice at a given iteration. 
    Places output in E[iter]. CUDA implementation.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model
        - lattice: CuArray array representing the spins in the lattice
        - E: Vector containing the total energy at all iterations
        - iter: Iteration at which we are computing the total energy
"""
function calcHamiltonian!(model::CUDAIsingModel,lattice,E,iter)
    N = model.N
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((N+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcHamiltonianKernel!(model,lattice,E,iter)
end
"""
    calcHamiltonianKernel!(model::CUDAIsing2DModel,lattice,E,iter)

    The kernel for calculating the total Energy for a 2D lattice at a 
    given iteration. Places output in E[iter]. CUDA implementation using 
    index reduction.

    Inputs:
        - model::CUDAIsing2DModel: The 2D CUDA Ising model
        - lattice: CuArray array representing the spins in the 2D lattice
        - E: Vector containing the total energy at all iterations
        - iter: Iteration at which we are computing the total energy
"""
function calcHamiltonianKernel!(model::CUDAIsing2DModel,lattice,E,iter)
    n = model.n
    J = model.J
    h = model.h
    n_threads = model.n_threads
    s = CuDynamicSharedArray(Float64,n_threads)

    tid = threadIdx().x
    start = tid + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x*gridDim().x
    stop = n^2-1
    s[tid] = 0.0;

    for idx ∈ start:stride:stop
        @inbounds i = Int64(floor((idx-1) / n))+1
        @inbounds j = Int64(mod((idx-1),n))+1
        @inbounds sl = lattice[(i-1)*n+(n+(j-2)%n)%n+1]
        @inbounds sr = lattice[(i-1)*n+(n+j%n)%n+1]
        @inbounds su = lattice[(n+i%n)%n*n+j]
        @inbounds sd = lattice[(n+(i-2)%n)%n*n+j]
        @inbounds sij = lattice[idx]

        @inbounds s[tid] -= sij*(J*(sl+sr+su+sd)/2+h)
    end
    sync_threads()

    j =  Int64(floor(blockDim().x/2))

    while j > 0
        if tid<j+1
            s[tid] += s[tid+j]
        end
        sync_threads()
        j = j>>1
    end

    if threadIdx().x==1
        CUDA.@atomic E[iter] += s[1]/((J*2+abs(h))*n^2)
    end
    return
end 
"""
    calcHamiltonianKernel!(model::CUDAIsing3DModel,lattice,E,iter)

    The kernel for calculating the total Energy for a 3D lattice at a 
    given iteration. Places output in E[iter]. CUDA implementation using 
    index reduction.

    Inputs:
        - model::CUDAIsing3DModel: The 3D CUDA Ising model
        - lattice: CuArray array representing the spins in the 3D lattice
        - E: Vector containing the total energy at all iterations
        - iter: Iteration at which we are computing the total energy
"""
function calcHamiltonianKernel!(model::CUDAIsing3DModel,lattice,E,iter)
    n = model.n
    J = model.J
    h = model.h
    n_threads = model.n_threads
    s = CuDynamicSharedArray(Float64,n_threads)

    tid = threadIdx().x
    start = tid + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x*gridDim().x
    stop = n^3-1
    s[tid] = 0.0;

    for idx ∈ start:stride:stop
        @inbounds i = Int64(floor((idx-1) / (n*n)))+1
        @inbounds j = Int64(floor(mod((idx-1)/n,n)))+1
        @inbounds k = Int64(mod((idx-1),n))+1

        @inbounds sl = lattice[(i-1)*n*n+(n+(j-2)%n)%n*n+(k-1)+1];
        @inbounds sr = lattice[(i-1)*n*n+(n+j%n)%n*n+(k-1)+1];
        @inbounds su = lattice[(n+i%n)%n*n*n+(j-1)*n+(k-1)+1];
        @inbounds sd = lattice[(n+(i-2)%n)%n*n*n+(j-1)*n+(k-1)+1];
        @inbounds sn = lattice[(i-1)*n*n+j*n+(n+(k-2)%n)%n+1];
        @inbounds ss = lattice[(i-1)*n*n+j*n+(n+k%n)%n+1];
        @inbounds sijk = lattice[idx];

        @inbounds s[tid] -= sijk*(J*(sl+sr+su+sd+sn+ss)/2+h)
    end
    sync_threads()

    j =  Int64(floor(blockDim().x/2))

    while j > 0
        if tid<j+1
            s[tid] += s[tid+j]
        end
        sync_threads()
        j = j>>1
    end

    if threadIdx().x==1
        CUDA.@atomic E[iter] += s[1]/((J*3+abs(h))*n^2)
    end
    return
end
"""
    calcMagnetisation!(model::CUDAIsingModel,lattice,m,iter)

    Calculates the net Magnetisation for a lattice at a given iteration. 
    Places output in m[iter]. CUDA implementation.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model
        - lattice: CuArray array representing the spins in the lattice
        - m: Vector containing the net magnetisation at all iterations
        - iter: Iteration at which we are computing the net magnetisation
"""
function calcMagnetisation!(model::CUDAIsingModel,lattice,m,iter)
    N = model.N
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((N+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcMagnetisationKernel!(model,lattice,m,iter)
end
"""
    calcMagnetisationKernel!(model::CUDAIsing2DModel,lattice,m,iter)

    The kernel for calculating the net Magnetisation for a lattice at a 
    given iteration. Places output in m[iter]. CUDA implementation using 
    index reduction.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model
        - lattice: CuArray array representing the spins in the 3D lattice
        - m: Vector containing the net magnetisation at all iterations
        - iter: Iteration at which we are computing the total energy
"""
function calcMagnetisationKernel!(model::CUDAIsingModel,lattice,m,iter)
    N = model.N
    J = model.J
    h = model.h
    n_threads = model.n_threads
    s = CuDynamicSharedArray(Float64,n_threads)

    tid = threadIdx().x
    start = tid + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x*gridDim().x
    stop = N-1
    s[tid] = 0.0;

    for idx ∈ start:stride:stop
        @inbounds s[tid] += lattice[idx]
    end

    sync_threads()

    j =  Int64(floor(blockDim().x/2))

    while j > 0
        if tid<j+1
            s[tid] += s[tid+j]
        end
        sync_threads()
        j = j>>1
    end

    if threadIdx().x==1
        CUDA.@atomic m[iter] += s[1]/N
    end
    return
end 
"""
    IsingIter!(model::CUDAIsingModel,lattice,rng)

    Executes a single iteration of the Metropolis Monte Carlo algorithm.
    Uses the 'checkerboard' method to flip spins which are independent 
    of each other simultaneously.

    Inputs:
        - model::CUDAIsingModel: The CUDA Ising model
        - lattice: CuArray representing the spins in the lattice
        - rng: Random number generator used in the Metropolis algorithms
"""
function IsingIter!(model::CUDAIsingModel,lattice,rng)
    N = model.N
    n_threads = model.n_threads
    n_blocks = Int64(floor((N+n_threads-1)/n_threads))

    rands = Random.rand(rng, Float64, N)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,false,lattice,rands)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,true,lattice,rands)
end
"""
    IsingIterKernel!(model::CUDAIsing2DModel,sublattice,lattice,rands)

    The kernel to perform the Metropolis Monte Carlo algorithm for the flip of a single
    spin in a 2D lattice. 

    Inputs:
        - model::CUDAIsing2DModel: The 2D Ising model
        - sublattice: Integer representing which spins we are flipping 
          (even or odd).
        - lattice: CuArray representing the spins in the 2D lattice
        - rng: Random number generator used in the Metropolis algorithms
"""
function IsingIterKernel!(model::CUDAIsing2DModel,sublattice,lattice,rands)
    n = model.n
    J = model.J 
    h = model.h

    tid = blockDim().x*(blockIdx().x-1)+threadIdx().x
    i = Int64(floor((tid-1) / n))+1
    j = Int64(mod((tid-1),n))+1

    if tid>=n^2
        return
    elseif ((i-1)%2 != (j-1)%2) != sublattice
        return
    end

    @inbounds sl = lattice[(i-1)*n+(n+(j-2)%n)%n+1]
    @inbounds sr = lattice[(i-1)*n+(n+j%n)%n+1]
    @inbounds su = lattice[(n+i%n)%n*n+j]
    @inbounds sd = lattice[(n+(i-2)%n)%n*n+j]
    @inbounds sij = lattice[tid];

    boltz = exp(-2*sij*((sl+sr+su+sd)*J+h))
    if (rands[tid]<=boltz)
        @inbounds lattice[tid]=-sij
    end
    return
end
"""
    IsingIterKernel!(model::CUDAIsing3DModel,sublattice,lattice,rands)

    The kernel to perform the Metropolis Monte Carlo algorithm for the flip of a single
    spin in a 3D lattice. 

    Inputs:
        - model::CUDAIsing3DModel: The 3D Ising model
        - sublattice: Integer representing which spins we are flipping 
          (even or odd).
        - lattice: CuArray representing the spins in the 3D lattice
        - rng: Random number generator used in the Metropolis algorithms
"""
function IsingIterKernel!(model::CUDAIsing3DModel,sublattice,lattice,rands)
    n = model.n
    J = model.J 
    h = model.h

    tid = blockDim().x*(blockIdx().x-1)+threadIdx().x
    i = Int64(floor((tid-1) / (n*n)))+1
    j = Int64(floor(mod((tid-1)/n,n)))+1
    k = Int64(mod((tid-1),n))+1
    
    if tid>=n^3
        return
    elseif ((k%2 == j%2) != i%2) != sublattice
        return
    end

    @inbounds sl = lattice[(i-1)*n*n+(n+(j-2)%n)%n*n+(k-1)+1];
    @inbounds sr = lattice[(i-1)*n*n+(n+j%n)%n*n+(k-1)+1];
    @inbounds su = lattice[(n+i%n)%n*n*n+(j-1)*n+(k-1)+1];
    @inbounds sd = lattice[(n+(i-2)%n)%n*n*n+(j-1)*n+(k-1)+1];
    @inbounds sn = lattice[(i-1)*n*n+j*n+(n+(k-2)%n)%n+1];
    @inbounds ss = lattice[(i-1)*n*n+j*n+(n+k%n)%n+1];
    @inbounds sijk = lattice[tid];

    boltz = exp(-2*sijk*((sl+sr+su+sd+sn+ss)*J+h))
    if (rands[tid]<=boltz)
        @inbounds lattice[tid]=-sijk
    end
    return
end