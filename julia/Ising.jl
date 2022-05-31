using LinearAlgebra

abstract type IsingModel end
abstract type Ising2DModel <: IsingModel end
abstract type Ising3DModel <: IsingModel end

struct Ising2DParam <: Ising2DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
end

struct Ising3DParam <: Ising3DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
end 

function MCIsing(model::IsingModel)
    n_iters = model.n_iters
    
    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)

    # Initialize matrix of spins (randomly assigned +-1)
    A,N = InitialiseIsing(model) 

    # Calculate initial m and E
    ms[1] = calcMagnetisation(model, A, N)
    Es[1] = calcHamiltonian(model, A)
    
    # Outer loop for MC iterations
    for l ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        IsingIter!(model,A)
        
        # Calculate observables
        ms[l] = calcMagnetisation(model,A, N)
        Es[l] = calcHamiltonian(model, A)
    end
    return ms, Es
end

function calcMagnetisation(model::IsingModel,A,N)
    return sum(A) / N
end

function InitialiseIsing(model::Ising2DModel)
    n = model.n
    return 1 .- 2 * rand([0, 1], (n,n)), n*n
end

function calcHamiltonian(model::Ising2DModel,A)
    E = 0.
    n = model.n
    J = model.J
    h = model.h
    for i ∈ 1:n, j ∈ 1:n
        E -= J*A[i,j]*(A[i,mod(n+mod(j,n),n)+1]+A[i,mod(n+mod(j-2,n),n)+1]+
                       A[mod(n+mod(i,n),n)+1,j]+A[mod(n+mod(i-2,n),n)+1,j])/2.0
        E -= h*A[i,j]
    end
    return E/((J*2+abs(h))*n^2)
end 

function calcDeltaHamiltonian(model::Ising2DModel,A,i,j)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*A[i,j]*(A[i,mod((n+mod(j,n)),n)+1]+A[i,mod((n+mod(j-2,n)),n)+1]+
                          A[mod((n+mod(i,n)),n)+1,j]+A[mod((n+mod(i-2,n)),n)+1,j])+
                          A[i,j]*h)
end

function IsingIter!(model::Ising2DModel,A)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n
        dE = calcDeltaHamiltonian(model, A, i, j)
        if dE <= 0 || rand() <= exp(-dE)
            A[i,j] *= -1
        end
    end
end

function InitialiseIsing(model::Ising3DModel)
    n = model.n
    return 1 .- 2 * rand([0, 1], (n,n,n)),n*n*n
end

function calcHamiltonian(model::Ising3DModel,A)
    E = 0.
    n = model.n
    J = model.J
    h = model.h
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        E -= J*A[i,j,k]*(A[i,mod(n+mod(j,n),n)+1,k]+A[i,mod(n+mod(j-2,n),n)+1,k]+
                         A[mod(n+mod(i,n),n)+1,j,k]+A[mod(n+mod(i-2,n),n)+1,j,k]+
                         A[i,j,mod(n+mod(k,n),n)+1]+A[i,j,mod(n+mod(k-2,n),n)+1])/2.0
        E -= h*A[i,j,k]
    end
    return E/((J*3+abs(h))*n^3)
end 

function calcDeltaHamiltonian(model::Ising3DModel,A,i,j,k)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*A[i,j,k]*(A[i,mod((n+mod(j,n)),n)+1,k]+A[i,mod((n+mod(j-2,n)),n)+1,k]+
                            A[mod((n+mod(i,n)),n)+1,j,k]+A[mod((n+mod(i-2,n)),n)+1,j,k]+
                            A[i,j,mod((n+mod(k,n)),n)+1]+A[i,j,mod((n+mod(k-2,n)),n)+1])+
                            A[i,j,k]*h)
end

function IsingIter!(model::Ising3DModel,A)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        dE = calcDeltaHamiltonian(model, A, i, j, k)
        if dE <= 0 || rand() <= exp(-dE)
            A[i,j,k] *= -1
        end
    end
end