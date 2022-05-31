include("../julia/Ising.jl")

using CUDA, Random
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
end

struct CUDAIsing3DParam <: CUDAIsing3DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    n_threads::Int64
end 

function init_lattice!(model::CUDAIsingModel,lattice,rands)
    n = model.n
    tid = blockDim().x*(blockIdx().x-1) + threadIdx().x
    if tid >= floor(n^2+1)
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

function MCIsing(model::CUDAIsingModel)
    n_threads = model.n_threads
    n_iters = model.n_iters

    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)

    lattice, rng, n_blocks = InitialiseIsing(model)
    ms[1] = calcMagnetisation(model,lattice)
    Es[1] = calcHamiltonian(model,lattice)

    for l ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        IsingIter!(model,lattice,rng,n_blocks)
        
        # Calculate observables
        ms[l] = calcMagnetisation(model, lattice)
        Es[l] = calcHamiltonian(model, lattice)
    end

    return ms, Es
end

function InitialiseIsing(model::CUDAIsing2DModel)
    n = model.n
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((n^2+n_threads-1)/n_threads))

    rng = CURAND.RNG(CURAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
    Random.seed!(rng,Int64(floor(time())))

    rands = Random.rand(rng, Float64, n^2)

    lattice = CuArray{Int}(undef,n^2)
    @cuda threads=n_threads blocks=n_blocks init_lattice!(model,lattice,rands)
    device_synchronize()
    return lattice, rng, n_blocks
end

function calcHamiltonian(model::CUDAIsing2DModel,lattice)
    E = CuArray(Float64[0])
    n = model.n
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((n^2+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcHamiltonian!(model,lattice,E)

    return Array(E)[1]
end

function calcHamiltonian!(model::CUDAIsing2DModel,lattice,E)
    n = model.n
    J = model.J
    h = model.h
    n_threads = model.n_threads
    s = CuDynamicSharedArray(Float64,n_threads)

    tid = threadIdx().x
    start = tid + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x*gridDim().x
    stop = n^2
    s[tid] = 0.0;

    for idx ∈ start:stride:stop
        i = Int64(floor((idx-1) / n))+1
        j = Int64(mod((idx-1),n))+1
        sl = lattice[(i-1)*n+(n+(j-2)%n)%n+1]
        sr = lattice[(i-1)*n+(n+j%n)%n+1]
        su = lattice[((n+i%n)%n)*n+j]
        sd = lattice[((n+(i-2)%n)%n)*n+j]

        # sl = lattice[i*n+(n+(j-2)%n)%n]
        # sr = lattice[i*n+(n+j%n)%n]
        # su = lattice[(n+i%n)%n*n+j]
        # sd = lattice[(n+(i-2)%n)%n*n+j]

        # sl = lattice[i*n+(n+(j-1)%n)%n]
        # sr = lattice[i*n+(n+(j+1)%n)%n]
        # su = lattice[(n+(i+1)%n)%n*n+j]
        # sd = lattice[(n+(i-1)%n)%n*n+j]

        sum_spins = sl+sr+su+sd
        sij = lattice[idx]
        s[tid] -= sij*(J*sum_spins/2+h)
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
        CUDA.atomic_add!(pointer(E),s[1]/((J*2+abs(h))*n^2))
    end
    return
end 

function calcMagnetisation(model::CUDAIsing2DModel,lattice)
    m = CuArray(Float64[0])
    n = model.n
    n_iters = model.n_iters
    n_threads = model.n_threads

    n_blocks = Int64(floor((n^2+n_threads-1)/n_threads))
    @cuda threads=n_threads blocks=n_blocks shmem=n_threads*sizeof(Float64) calcMagnetisation!(model,lattice,m)
    return Array(m)[1]
end

function calcMagnetisation!(model::CUDAIsing2DModel,lattice,m)
    n = model.n
    J = model.J
    h = model.h
    n_threads = model.n_threads
    s = CuDynamicSharedArray(Float64,n_threads)

    tid = threadIdx().x
    start = tid + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x*gridDim().x
    stop = n^2
    s[tid] = 0.0;

    for idx ∈ start:stride:stop
        s[tid] += lattice[idx]
    end

    sync_threads()

    j =  Int64(floor(blockDim().x/2))

    while j > 0
        if tid<j
            s[tid] += s[tid+j]
        end
        sync_threads()
        j = j>>1
    end

    if threadIdx().x==1
        CUDA.atomic_add!(pointer(m),s[1]/n^2)
    end
    return
end 

function IsingIter!(model::CUDAIsing2DModel,lattice,rng, n_blocks)
    n = model.n
    n_threads = model.n_threads
    rands = Random.rand(rng, Float64, n^2)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,0,lattice,rands)
    @cuda threads=n_threads blocks=n_blocks IsingIterKernel!(model,1,lattice,rands)
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

    sl = lattice[(i-1)*n+(n+(j-2)%n)%n+1]
    sr = lattice[(i-1)*n+(n+j%n)%n+1]
    su = lattice[(n+i%n)%n*n+j]
    sd = lattice[(n+(i-2)%n)%n*n+j]

    sum_spins = sl+sr+su+sd
    s = lattice[tid]
    boltz = exp(-2*s*(sum_spins*J+h))
    if (rands[tid]<=boltz)
        lattice[tid]=-s
    end
    return
end