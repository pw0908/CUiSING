# Multiple Dispatch
One of Julia's most-powerful features is multiple dispatching. To understand what this is, one must first understand how types work in Julia. For example:
```julia
julia> a=1
1

julia> typeof(a)
Int64

julia> b=1.
1.0

julia> typeof(b)
Float64

julia> c="hello world!"
"hello world!"

julia> typeof(c)
String
```
As we can see, Julia has numerous type definitions. These types all belong to a hierarchy of types shown below:
![Julia Types](assets/Type.png)
The difference between abstract and concrete types is that concrete types are instances of abstract types. 

The reason types are so important in Julia is because functions can be defined such that they can only accept inputs of a certain type. For example:
```julia
function foo(x::Int64)
    return x+1
end

function foo(x::Number)
    return x^2
end

function foo(x::Any)
    return x*x
end
```
We can then perform the following:
```julia
julia> foo(a)
2

julia> foo(b)
1.0

julia> foo(c)
"hello world!hello world!"
```

As we can see, calling `foo(a)`, as it is an integer, used the first definition of our function. However, calling `foo(b)`, instead of producing an error as no Float64 based functions were provided, julia looked for the next most-specific function for our input. As `Float64` is a sub-type of `Number`, it used our second function definition. Finally, as `c` is a `String`, the only function that it can be applied to is our last definition which used the most-general type, `Any`. 

As one can imagine, this feature can be used to define functions which are optimised for a given input type. For example, if a matrix is of type `Diagonal`, one could use a faster algorithm to compute its determinant than if it were a regular `Array`.

## Custom types
Mutliple dispatch truly becomes a powerful tool with the ability to define custom types. In the context of CUiSING, we have defined new abstract types:
```julia
abstract type IsingModel end
abstract type Ising2DModel <: IsingModel end
abstract type Ising3DModel <: IsingModel end

struct Ising2DParam <: Ising2DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    N::Int64
end

struct Ising3DParam <: Ising3DModel
    n::Int64
    J::Float64
    h::Float64
    n_iters::Int64
    N::Int64
end 
```

In the above, we have effectively defined structs for 2D and 3D Ising simulations which are of type `Ising2DModel` and `Ising3DModel`, respectively. Both are sub-types of `IsingModel`. What this lets us do is define some functions which are both generalised to both simulations, for example, the actual execution of the simulation `MCIsing`:
```julia
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
```
On the other hand, we can define functions specific to a particular simulation type. For example, `IsingIter!`:
```julia
function IsingIter!(model::Ising2DModel,lattice,rng)
    n = model.n
    for i ∈ 1:n, j ∈ 1:n
        dE = calcDeltaHamiltonian(model, lattice, i, j)
        if dE <= 0 || rand(rng) <= exp(-dE)
            lattice[i,j] *= -1
        end
    end
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
```
One can think of this as Julia's response to object-oriented programming. This allows for very concise and clean code. For example, when developing the CUDA implementation of the code, new types were defined:
```julia
abstract type CUDAIsingModel <: IsingModel end
abstract type CUDAIsing2DModel <: CUDAIsingModel end
abstract type CUDAIsing3DModel <: CUDAIsingModel end
```
It was unnecessary to re-define `MCIsing` for CUDA-specific implementations as `CUDAIsingModel` was defined as a sub-type of `IsingModel`!
