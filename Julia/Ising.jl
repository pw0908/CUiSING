using LinearAlgebra

function findmat2d(f, A::AbstractMatrix)
    m,n = size(A)
    row = Int[]
    col = Int[]
    val = Int[]
    for i in 1:m, j in 1:n
      f(A[i,j]) && push!(row,i); push!(col,j); push!(val,A[i,j]); 
    end
    row,col,val
end

# function diff_neigh2d(xy::Array, A::Array, n::Int)
#     return  -sum( hcat(A[xy[:,1]], A[mod.(xy[:, 1], n) .+ 1 + n * (xy[:, 2] .- 1)],
#                        A[xy[:, 1] + n * mod.(xy[:, 2] .- 2, n)], A[xy[:, 1] + n * mod.(xy[:, 2], n)]);dims=2);
# end

function calcHamiltonian2d(n,J,h,A)
    E = 0.
    for i ∈ 1:n, j ∈ 1:n
        E -= J*A[i,j]*(A[i,mod(n+mod(j,n),n)+1]+A[i,mod(n+mod(j-2,n),n)+1]+
                       A[mod(n+mod(i,n),n)+1,j]+A[mod(n+mod(i-2,n),n)+1,j])/2.0
        E -= h*A[i,j]
    end
    return E
end 

function calcDeltaHamiltonian2d(n,J,h,A,i,j)
    return 2.0*(J*A[i,j]*(A[i,mod((n+mod(j,n)),n)+1]+A[i,mod((n+mod(j-2,n)),n)+1]+
                          A[mod((n+mod(i,n)),n)+1,j]+A[mod((n+mod(i-2,n)),n)+1,j])+
                          A[i,j]*h);
end

function MCIsing2d(n,J,h,n_iters;sample_freq=20)
    N = n^2;
    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1); Es = deepcopy(ms);

    # Initialize matrix of spins (randomly assigned +-1)
    A = 1 .- 2 * rand([0, 1], (n,n));

    # Store spin values as row, col, val
    (row,col,val) = findmat2d(!iszero,A);
    pos = hcat(row, col, val);

    # Calculate initial m and E
    ms[1] = sum(A) / N;
    Es[1] = calcHamiltonian2d(n, J, h, A);
    
    # Outer loop for MC iterations
    for i ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        for j ∈ 1:N
            dE = calcDeltaHamiltonian2d(n, J, h, A, pos[j,1], pos[j,2]);
            if dE <= 0 || rand() <= exp(-dE)
                A[pos[j, 1], pos[j, 2]] = -pos[j, 3];
                pos[j,3] *= -1
            end
        end
        
        # Calculate observables
        ms[i] = sum(A) / N;
        Es[i] = calcHamiltonian2d(n, J, h, A);
    end
    return ms, Es
end

# function findmat3d(f, A::AbstractMatrix)
#     m,n,l = size(A)
#     row = Int[]
#     col = Int[]
#     wid = Int[]
#     val = Int[]
#     for i in 1:m, j in 1:n, k in 1:l
#       f(A[i,j,k]) && push!(row,i); push!(col,j); push!(wid,k); push!(val,A[i,j,k]); 
#     end
#     row,col,wid,val
# end

# function diff_neigh3d(xyz::Array, A::Array, n::Int)
#     return  -sum( hcat(A[xyz[:,1]], A[mod.(xyz[:, 1], n) .+ 1 + n * (xyz[:, 2] .- 1)],
#                        A[xyz[:, 1] + n * mod.(xy[:, 2] .- 2, n)], A[xy[:, 1] + n * mod.(xy[:, 2], n)]);dims=2);
# end

# function MCIsing3d(n,J,h,n_iters)
#     N = n^3;
#     # Initialize the vectors for storing magnetization and energy
#     ms = zeros(n_iters + 1, 1); Es = ms;

#     # Initialize matrix of spins (randomly assigned +-1)
#     A = 1 .- 2 * rand([0, 1], (n,n,n));

#     # Store spin values as row, col, val
#     (row,col,wid,val) = findmat3d(!iszero,A);
#     pos = hcat(row, col, wid, val);

#     # Calculate initial m and E
#     ms[1] = sum(A) / N;
#     Es[1] = -J * dot(pos[:, 4],diff_neigh3d(pos, A, n))/2;

#     # Outer loop for MC iterations
#     for i ∈ 2:n_iters + 1
#         # In each MC iteration, attempt to flip all spins, using metropolis
#         for j ∈ 1:N
#             dE = 2 * J * pos[j, 4] * (A[mod(pos[j, 1] - 2, n) + 1, pos[j, 2], pos[j, 3]]+ A[mod(pos[j, 1], n) + 1, pos[j, 2], pos[j, 3]]+
#                                       A[pos[j, 1], mod(pos[j, 2] - 2, n) + 1, pos[j, 3]]+ A[pos[j, 1], mod(pos[j, 2], n) + 1, pos[j, 3]]+
#                                       A[pos[j, 1], pos[j,2], mod(pos[j , 3] - 2, n) + 1]+ A[pos[j, 1], pos[j, 2], mod(pos[j, 3], n) + 1]) 
#                                       + h * pos[j,4];
#             if dE <= 0 || rand() <= exp(-dE)
#                 A[pos[j, 1], pos[j, 2], pos[j, 3]] = -pos[j, 4];
#                 pos[j,4] *= -1
#             end
#         end
        
#         # Calculate observables
#         ms[i] = sum(A) / N;
#         Es[i] = J * dot(pos[:, 4], diff_neigh(pos, A, n))/2;
#     end
#     return ms, Es
# end