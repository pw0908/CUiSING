using LinearAlgebra, PyCall
import PyPlot; const plt=PyPlot

# Set x and y dim (n), and number of spins n^2
n = 10; N = 1;

# J is the interaction strength
# Sample freq sets how frequently to print the visualization
J = 0.5; n_iters = 10^4; sample_freq = 20;

# Initialize the vectors for storing magnetization and energy
ms = zeros(n_iters + 1, 1); Es = ms;

# Initialize matrix of spins (randomly assigned +-1)
A = 1 .- 2 * rand([0, 1], (n,n));

function findmat(f, A::AbstractMatrix)
    m,n = size(A)
    row = Int[]
    col = Int[]
    val = Int[]
    for i in 1:m, j in 1:n
      f(A[i,j]) && push!(row,i); push!(col,j); push!(val,A[i,j]); 
    end
    row,col,val
end

# Store spin values as row, col, val
(row,col,val) = findmat(!iszero,A);
pos = hcat(row, col, val);

function diff_neigh(xy, A, n)
    return  -sum( hcat(A[mod.(xy[:, 1] .- 2, n) .+ 1 + n * (xy[:, 2] .- 1)],
        A[mod.(xy[:, 1], n) .+ 1 + n * (xy[:, 2] .- 1)],
        A[xy[:, 1] + n * mod.(xy[:, 2] .- 2, n)],
        A[xy[:, 1] + n * mod.(xy[:, 2], n)]);dims=2);
end

# Calculate initial m and E
ms[1] = sum(A) / N;
Es[1] = -J * dot(pos[:, 3],diff_neigh(pos, A, n))/2;

# Outer loop for MC iterations
for i ∈ 2:n_iters + 1
    # In each MC iteration, attempt to flip all spins, using metropolis
    for j ∈ 1:N
        dE = 2 * J * pos[j, 3] * (A[mod(pos[j, 1] - 2, n) + 1, pos[j, 2]]+
            A[mod(pos[j, 1], n) + 1, pos[j, 2]]+
            A[pos[j, 1], mod(pos[j, 2] - 2, n) + 1]+
            A[pos[j, 1], mod(pos[j, 2], n) + 1]);
        if dE <= 0 || rand() <= exp(-dE)
            A[pos[j, 1], pos[j, 2]] = -pos[j, 3];
            pos[j,3] *= -1
        end
    end
    
    # Calculate observables
    ms[i] = sum(A) / N;
    Es[i] = J * dot(pos[:, 3], diff_neigh(pos, A, n))/2;
end

plt.plot(Es)
plt.savefig("trial.pdf")
