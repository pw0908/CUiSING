using PyCall
import PyPlot; const plt=PyPlot

include("Ising.jl")

n = 100
J = 1.0
h = 0.
n_iters = 1000
n_threads = 256
iter = 1:n_iters+1
precompile(MCIsing,(CUDAIsing2DParam,))
model = CUDAIsing2DParam(n, J, h, n_iters,n_threads)

M, E = @time MCIsing(model)

model = Ising2DParam(n, J, h, n_iters)
M1,E1 = @time MCIsing(model)

plt.clf()
plt.plot(iter,E)
plt.plot(iter,E1)
plt.savefig("test.png")