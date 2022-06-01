using LinearAlgebra

abstract type IsingModel end
abstract type Ising2DModel <: IsingModel end
abstract type Ising3DModel <: IsingModel end

struct Ising2DParam <: Ising2DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    N::Int64
    function Ising2DParam(n::Int64,J::Float64,h::Float64,
        n_iters::Int64)
        return new(n,J,h,n_iters,n^2)
    end
end

struct Ising3DParam <: Ising3DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    N::Int64
    function Ising3DParam(n::Int64,J::Float64,h::Float64,
        n_iters::Int64)
        return new(n,J,h,n_iters,n^3)
    end
end 

function MCIsing(model::IsingModel)
    n_iters = model.n_iters

    # Initialize matrix of spins (randomly assigned +-1)
    lattice,rng,ms,Es = InitialiseIsing(model) 

    # Calculate initial m and E
    calcMagnetisation!(model, lattice, ms, 1)
    calcHamiltonian!(model, lattice, Es, 1)
    
    # Outer loop for MC iterations
    for l ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        IsingIter!(model,lattice,rng)
        
        # Calculate observables
        calcMagnetisation!(model, lattice, ms, l)
        calcHamiltonian!(model, lattice, Es, l)
    end
    return ms, Es
end

function calcMagnetisation!(model::IsingModel,lattice,m,iter)
    m[iter] = sum(lattice) / model.N
end

function InitialiseIsing(model::Ising2DModel)
    n_iters = model.n_iters
    n = model.n

    rng = Xoshiro(Int64(floor(time())))

    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)
    return 1 .- 2 * rand([0, 1], (n,n)), rng, ms, Es
end

function calcHamiltonian!(model::Ising2DModel,lattice,E,iter)
    Es = 0.
    n = model.n
    J = model.J
    h = model.h
    for i ∈ 1:n, j ∈ 1:n
        Es -= J*lattice[i,j]*(lattice[i,mod(n+mod(j,n),n)+1]+lattice[i,mod(n+mod(j-2,n),n)+1]+
                       lattice[mod(n+mod(i,n),n)+1,j]+lattice[mod(n+mod(i-2,n),n)+1,j])/2.0
        Es -= h*lattice[i,j]
    end
    E[iter] = Es/((J*2+abs(h))*n^2)
end 

function calcDeltaHamiltonian(model::Ising2DModel,lattice,i,j)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*lattice[i,j]*(lattice[i,mod((n+mod(j,n)),n)+1]+lattice[i,mod((n+mod(j-2,n)),n)+1]+
                          lattice[mod((n+mod(i,n)),n)+1,j]+lattice[mod((n+mod(i-2,n)),n)+1,j])+
                          lattice[i,j]*h)
end

function IsingIter!(model::Ising2DModel,lattice,rng)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n
        dE = calcDeltaHamiltonian(model, lattice, i, j)
        if dE <= 0 || rand(rng) <= exp(-dE)
            lattice[i,j] *= -1
        end
    end
end

function InitialiseIsing(model::Ising3DModel)
    n_iters = model.n_iters
    n = model.n

    rng = Xoshiro(Int64(floor(time())))

    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)
    return 1 .- 2 * rand([0, 1], (n,n,n)), rng, ms, Es
end

function calcHamiltonian(model::Ising3DModel,lattice)
    E = 0.
    n = model.n
    J = model.J
    h = model.h
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        E -= J*lattice[i,j,k]*(lattice[i,mod(n+mod(j,n),n)+1,k]+lattice[i,mod(n+mod(j-2,n),n)+1,k]+
                         lattice[mod(n+mod(i,n),n)+1,j,k]+lattice[mod(n+mod(i-2,n),n)+1,j,k]+
                         lattice[i,j,mod(n+mod(k,n),n)+1]+lattice[i,j,mod(n+mod(k-2,n),n)+1])/2.0
        E -= h*lattice[i,j,k]
    end
    return E/((J*3+abs(h))*n^3)
end 

function calcDeltaHamiltonian(model::Ising3DModel,lattice,i,j,k)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*lattice[i,j,k]*(lattice[i,mod((n+mod(j,n)),n)+1,k]+lattice[i,mod((n+mod(j-2,n)),n)+1,k]+
                            lattice[mod((n+mod(i,n)),n)+1,j,k]+lattice[mod((n+mod(i-2,n)),n)+1,j,k]+
                            lattice[i,j,mod((n+mod(k,n)),n)+1]+lattice[i,j,mod((n+mod(k-2,n)),n)+1])+
                            lattice[i,j,k]*h)
end

function IsingIter!(model::Ising3DModel,lattice,rng)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        dE = calcDeltaHamiltonian(model, lattice, i, j, k)
        if dE <= 0 || rand(rng) <= exp(-dE)
            lattice[i,j,k] *= -1
        end
    end
end