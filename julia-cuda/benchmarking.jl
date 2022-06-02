include("Ising.jl")
using Printf

n = parse(Int64, ARGS[3]);
J = parse(Float64, ARGS[4]); n_iters = parse(Int64, ARGS[1]); h= parse(Float64, ARGS[5]);
d = parse(Int64, ARGS[2])
n_threads = 256
iter = 1:n_iters+1
precompile(MCIsing,(CUDAIsing2DParam,))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100000
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

t = []
for i in 1:length(n)
    T = let model = CUDAIsing2DParam(n[i], J, h, n_iters, n_threads)
    b = @benchmarkable MCIsing($model) samples=5 evals=1 seconds=10000
    run(b)
    end
    append!(t,mean(T.times)*1e-9)
    
    print(i," ",n[i]," ",t[i],"\n")
end
MCIsing(model)
M, E = @btime MCIsing(model)

# println(t)

open("benchmarking/2d/data/julia_gpu_output.dat", "w") do f
    for i = 1:length(n)
        println(f, n[i]," ",@sprintf("%.5f",t[i]))
    end
end
