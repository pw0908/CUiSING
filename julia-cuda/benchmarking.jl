using PyCall
import PyPlot; const plt=PyPlot

include("Ising.jl")

n = 100
J = 0.1
h = 0.
n_iters = 1000
n_threads = 256
iter = 1:n_iters+1
precompile(MCIsing,(CUDAIsing2DParam,))
model = CUDAIsing2DParam(n, J, h, n_iters,n_threads)

println("GPU:")
MCIsing(model)
M, E = @time MCIsing(model)

model = Ising2DParam(n, J, h, n_iters)
println("CPU:")
M1,E1 = @time MCIsing(model)

plt.clf()
plt.plot(iter,M)
plt.plot(iter,M1)
plt.savefig("test.png")