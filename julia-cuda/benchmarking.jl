include("Ising.jl")
using Printf, BenchmarkTools

J = 0.1
h = 0.
n_iters = 1000
d = 2
n = Int64.(unique(2*round.(10 .^range(0.5,3.5,60))))
n_threads = 256
iter = 1:n_iters+1
precompile(MCIsing,(CUDAIsing2DParam,))

t = []
for i in 1:length(n)
    T = let model = CUDAIsing2DParam(n[i], J, h, n_iters, n_threads)
    @benchmark MCIsing($model)
    end
    append!(t,mean(T.times)*1e-9)
end

open("benchmarking/2d/data/julia_gpu_output.dat", "w") do f
    for i = 1:length(n)
        println(f, i," ",@sprintf("%.5f",t[i]))
    end
end