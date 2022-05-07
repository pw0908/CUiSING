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

function findmat3d(f, A::Array)
    m,n,l = size(A)
    row = Int[]
    col = Int[]
    wid = Int[]
    val = Int[]
    for i in 1:m, j in 1:n, k in 1:l
      f(A[i,j,k]) && push!(row,i); push!(col,j); push!(wid,k); push!(val,A[i,j,k]); 
    end
    row,col,wid,val
end

function calcHamiltonian3d(n,J,h,A)
    E = 0.
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n
        E -= J*A[i,j,k]*(A[i,mod(n+mod(j,n),n)+1,k]+A[i,mod(n+mod(j-2,n),n)+1,k]+
                         A[mod(n+mod(i,n),n)+1,j,k]+A[mod(n+mod(i-2,n),n)+1,j,k]+
                         A[i,j,mod(n+mod(k,n),n)+1]+A[i,j,mod(n+mod(k-2,n),n)+1])/2.0
        E -= h*A[i,j,k]
    end
    return E
end 

function calcDeltaHamiltonian3d(n,J,h,A,i,j,k)
    return 2.0*(J*A[i,j,k]*(A[i,mod((n+mod(j,n)),n)+1,k]+A[i,mod((n+mod(j-2,n)),n)+1,k]+
                            A[mod((n+mod(i,n)),n)+1,j,k]+A[mod((n+mod(i-2,n)),n)+1,j,k]+
                            A[i,j,mod((n+mod(k,n)),n)+1]+A[i,j,mod((n+mod(k-2,n)),n)+1])+
                            A[i,j,k]*h);
end

function MCIsing3d(n,J,h,n_iters;sample_freq=20)
    N = n^3;
    # Initialize the vectors for storing magnetization and energy
    ms = zeros(n_iters + 1, 1); Es = deepcopy(ms);

    # Initialize matrix of spins (randomly assigned +-1)
    A = 1 .- 2 * rand([0, 1], (n,n,n));

    # Store spin values as row, col, val
    (row,col,wid,val) = findmat3d(!iszero,A);
    pos = hcat(row, col, wid, val);

    # Calculate initial m and E
    ms[1] = sum(A) / N;
    Es[1] = calcHamiltonian3d(n, J, h, A);
    
    # Outer loop for MC iterations
    for i ∈ 2:n_iters + 1
        # In each MC iteration, attempt to flip all spins, using metropolis
        for j ∈ 1:N
            dE = calcDeltaHamiltonian3d(n, J, h, A, pos[j,1], pos[j,2], pos[j,3]);
            if dE <= 0 || rand() <= exp(-dE)
                A[pos[j, 1], pos[j, 2], pos[j, 3]] = -pos[j, 4];
                pos[j, 4] *= -1
            end
        end
        
        # Calculate observables
        ms[i] = sum(A) / N;
        Es[i] = calcHamiltonian3d(n, J, h, A);
    end
    return ms, Es
end