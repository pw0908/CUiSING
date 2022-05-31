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
    return E/(J*2*n^2)
end 

function calcDeltaHamiltonian(model::Ising2DModel,A,i,j)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*A[i,j]*(A[i,mod((n+mod(j,n)),n)+1]+A[i,mod((n+mod(j-2,n)),n)+1]+
                          A[mod((n+mod(i,n)),n)+1,j]+A[mod((n+mod(i-2,n)),n)+1,j])+
                          A[i,j]*h)
end

function MCIsing(model::Ising2DModel)
    n = model.n
    n_iters = model.n_iters
    N = n^2
    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)

    # Initialize matrix of spins (randomly assigned +-1)
    A = 1 .- 2 * rand([0, 1], (n,n))

    # Calculate initial m and E
    ms[1] = sum(A) / N
    Es[1] = calcHamiltonian(model, A)
    
    # Outer loop for MC iterations
    for l ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        for i ∈ 1:n, j ∈ 1:n
            dE = calcDeltaHamiltonian(model, A, i, j)
            if dE <= 0 || rand() <= exp(-dE)
                A[i,j] *= -1
            end
        end
        
        # Calculate observables
        ms[l] = sum(A) / N
        Es[l] = calcHamiltonian(model, A)
    end
    return ms, Es
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
    return E/(J*3*n^3)
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

function MCIsing(model::Ising3DModel)
    n = model.n
    n_iters = model.n_iters
    N = n^3
    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)

    # Initialize matrix of spins (randomly assigned +-1)
    A = 1 .- 2 * rand([0, 1], (n,n,n))

    # Calculate initial m and E
    ms[1] = sum(A) / N
    Es[1] = calcHamiltonian(model, A)
    
    # Outer loop for MC iterations
    for l ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
            dE = calcDeltaHamiltonian(model, A, i, j, k)
            if dE <= 0 || rand() <= exp(-dE)
                A[i,j,k] *= -1
            end
        end
        
        # Calculate observables
        ms[l] = sum(A) / N
        Es[l] = calcHamiltonian(model, A)
    end
    return ms, Es
end