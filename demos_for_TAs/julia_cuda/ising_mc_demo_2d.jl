# Import modules
using PyCall
import PyPlot; const plt=PyPlot

# Set figure parameters
WIDTH = 1.5 * 8.3 / 2.54
DOUBLE_WIDTH = 1.5 * 17.1 / 2.54
DPI = 350
format="png"

# Include the necessary code
include("../../julia-cuda/Ising.jl")

# Input parameters (d=2 by default here)
n_iters = Int64(1e4)
n_threads = 256
n = 300
J = 0.5
h = 0.0

# Build the CUDA Ising model
model = CUDAIsing2DParam(n,J,h,n_iters,n_threads)
# model = Ising2DParam(n,J,h,n_iters)

# Run the cpp-cuda code with the given parameters
Ms, Es, lattice = MCIsing(model)
iters = 1:n_iters+1
lattice = reshape(lattice, n, n)


plt.figure(1,tight_layout=true)
plt.clf()
plt.plot(iters,Es,"-k",label="E")
plt.plot(iters,Ms,"-r",label="M")
plt.xlabel("Iters")
plt.title("Trajectories")
plt.savefig("./figures/trajectories.png",bbox_inches="tight")

plt.figure(2,tight_layout=true)
plt.clf()
plt.title("Lattice")
plt.imshow(lattice,cmap="gray")
plt.colorbar(ticks=[-1,1])
plt.savefig("./figures/lattice.png",bbox_inches="tight")