using LinearAlgebra, Random

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


"""
    MCIsing(model::IsingModel)

    Runs the whole monte carlo simulation for all Ising models.

    Inputs:
        - model::IsingModel: Any Ising model 
    Outputs:
        - ms: Net magnetisation at each iteration
        - Es: Total energy at each iteration
        - lattice: An array representing the spins in the lattice
"""
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
    return ms, Es, lattice
end

"""
    calcMagnetisation!(model::IsingModel,lattice,m,iter)

    Calculates the net Magnetisation for a lattice at a 
    given iteration. Places output in m[iter].

    Inputs:
        - model::Ising2DModel: The Ising model (2D and 3D)
        - lattice: 2D or 3D array representing the spins in the lattice
        - m: Vector containing the net magnetisation at all iterations
        - iter: Iteration at which we are computing the net magnetisation
"""

function calcMagnetisation!(model::IsingModel,lattice,m,iter)
    m[iter] = sum(lattice) / model.N
end

"""
    InitialiseIsing(model::Ising2DModel)

    Initialise the Ising model for a 2D lattice.

    Inputs:
        - model::Ising2DModel: The 2D Ising model 
    Outputs:
        - lattice: 2D array representing the spins in the lattice
        - rng: Random number generator used in the Metropolis algorithms
        - ms: Net magnetisation at each iteration
        - Es: Total energy at each iteration
"""
function InitialiseIsing(model::Ising2DModel)
    n_iters = model.n_iters
    n = model.n

    rng = Random.Xoshiro(Int64(floor(time())))

    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)
    return 1 .- 2 * rand([0, 1], (n,n)), rng, ms, Es
end

"""
    calcHamiltonian!(model::Ising2DModel,lattice,E,iter)

    Calculates the total Energy for a 2D lattice at a 
    given iteration. Places output in E[iter].

    Inputs:
        - model::Ising2DModel: The 2D Ising model
        - lattice: 2D array representing the spins in the lattice
        - E: Vector containing the total energy at all iterations
        - iter: Iteration at which we are computing the total energy
"""
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

"""
    calcDeltaHamiltonian(model::Ising2DModel,lattice,i,j)

    Calculates the Energy change in a 2D lattice when a spin at [i,j] 
    is flipped. Places output in E[iter].

    Inputs:
        - model::Ising2DModel: The 2D Ising model
        - lattice: 2D array representing the spins in the lattice
        - i: x-coordinate of the spin being flipped
        - j: y-coordinate of the spin being flipped
    Output:
        - dE: Energy change when a spin at [i,j] is flipped
"""
function calcDeltaHamiltonian(model::Ising2DModel,lattice,i,j)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*lattice[i,j]*(lattice[i,mod((n+mod(j,n)),n)+1]+lattice[i,mod((n+mod(j-2,n)),n)+1]+
                          lattice[mod((n+mod(i,n)),n)+1,j]+lattice[mod((n+mod(i-2,n)),n)+1,j])+
                          lattice[i,j]*h)
end

"""
    IsingIter!(model::Ising2DModel,lattice,rng)

    Executes a single iteration of the Metropolis Monte Carlo algorithm.
    Obtains the energy change for flipping each spin in the lattice, 
    evaluates the probability and determines whether or not to accept the
    change.

    Inputs:
        - model::Ising2DModel: The 2D Ising model
        - lattice: 2D array representing the spins in the lattice
        - Random number generator used in the Metropolis algorithms
"""
function IsingIter!(model::Ising2DModel,lattice,rng)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n
        dE = calcDeltaHamiltonian(model, lattice, i, j)
        if dE <= 0 || rand(rng) <= exp(-dE)
            lattice[i,j] *= -1
        end
    end
end

"""
    InitialiseIsing(model::Ising3DModel)

    Initialise the Ising model for a 3D lattice.

    Inputs:
        - model::Ising3DModel: The 3D Ising model 
    Outputs:
        - lattice: 3D array representing the spins in the lattice
        - rng: Random number generator used in the Metropolis algorithms
        - ms: Net magnetisation at each iteration
        - Es: Total energy at each iteration
"""
function InitialiseIsing(model::Ising3DModel)
    n_iters = model.n_iters
    n = model.n

    rng = Xoshiro(Int64(floor(time())))

    ms = zeros(n_iters + 1, 1) 
    Es = zeros(n_iters + 1, 1)
    return 1 .- 2 * rand([0, 1], (n,n,n)), rng, ms, Es
end

"""
    calcHamiltonian!(model::Ising3DModel,lattice,E,iter)

    Calculates the total Energy for a 3D lattice at a 
    given iteration. Places output in E[iter].

    Inputs:
        - model::Ising3DModel: The 3D Ising model
        - lattice: 3D array representing the spins in the lattice
        - E: Vector containing the total energy at all iterations
        - iter: Iteration at which we are computing the total energy
"""
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

"""
    calcDeltaHamiltonian(model::Ising2DModel,lattice,i,j)

    Calculates the Energy change in a 3D lattice when a spin at [i,j,k] 
    is flipped. Places output in E[iter].

    Inputs:
        - model::Ising3DModel: The 3D Ising model
        - lattice: 3D array representing the spins in the lattice
        - i: x-coordinate of the spin being flipped
        - j: y-coordinate of the spin being flipped
        - k: z-coordinate of the spin being flipped
    Output:
        - dE: Energy change when a spin at [i,j,k] is flipped
"""
function calcDeltaHamiltonian(model::Ising3DModel,lattice,i,j,k)
    n = model.n
    J = model.J
    h = model.h
    return 2.0*(J*lattice[i,j,k]*(lattice[i,mod((n+mod(j,n)),n)+1,k]+lattice[i,mod((n+mod(j-2,n)),n)+1,k]+
                            lattice[mod((n+mod(i,n)),n)+1,j,k]+lattice[mod((n+mod(i-2,n)),n)+1,j,k]+
                            lattice[i,j,mod((n+mod(k,n)),n)+1]+lattice[i,j,mod((n+mod(k-2,n)),n)+1])+
                            lattice[i,j,k]*h)
end
"""
    IsingIter!(model::Ising3DModel,lattice,rng)

    Executes a single iteration of the Metropolis Monte Carlo algorithm.
    Obtains the energy change for flipping each spin in the lattice, 
    evaluates the probability and determines whether or not to accept the
    change.

    Inputs:
        - model::Ising3DModel: The 3D Ising model
        - lattice: 3D array representing the spins in the lattice
        - Random number generator used in the Metropolis algorithms
"""
function IsingIter!(model::Ising3DModel,lattice,rng)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        dE = calcDeltaHamiltonian(model, lattice, i, j, k)
        if dE <= 0 || rand(rng) <= exp(-dE)
            lattice[i,j,k] *= -1
        end
    end
end