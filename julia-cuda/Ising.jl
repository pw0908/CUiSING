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

function calcHamiltonian!(model::CUDAIsingModel,lattice,E,iter)
    N = model.N
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((N+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcHamiltonianKernel!(model,lattice,E,iter)
end

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

function calcMagnetisation!(model::CUDAIsingModel,lattice,m,iter)
    N = model.N
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((N+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcMagnetisationKernel!(model,lattice,m,iter)
end

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

function IsingIter!(model::CUDAIsingModel,lattice,rng)
    N = model.N
    n_threads = model.n_threads
    n_blocks = Int64(floor((N+n_threads-1)/n_threads))

    rands = Random.rand(rng, Float64, N)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,false,lattice,rands)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,true,lattice,rands)
end

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