using Test, BenchmarkTools, PyCall
import PyPlot; const plt=PyPlot
WIDTH = 1.5 * 8.3 / 2.54
DOUBLE_WIDTH = 1.5 * 17.1 / 2.54
DPI = 350
format="png"

# rcParams = PyDict(matplotlib["rcParams"])
# rcParams["axes.labelsize"] = 14
# rcParams["axes.xmargin"] = 0
# rcParams["axes.ymargin"] = .1
# rcParams["lines.markersize"] = 3
# rcParams["figure.dpi"] = DPI
# rcParams["figure.autolayout"] = true
# rcParams["figure.figsize"] = (WIDTH, 3 * WIDTH / 4)
# rcParams["figure.facecolor"] = "white"
# rcParams["font.size"] = 12
# rcParams["grid.color"] = "0"
# rcParams["grid.linestyle"] = "-"
# rcParams["legend.edgecolor"] = "1"
# rcParams["legend.fontsize"] = 12
# rcParams["xtick.labelsize"] = 12
# rcParams["ytick.labelsize"] = 12
# rcParams["font.family"] = "DeJavu Serif"
# rcParams["font.serif"] = "Computer Modern Roman"
# rcParams["mathtext.fontset"] = "cm"
# rcParams["mathtext.rm"] = "serif"
# rcParams["text.usetex"] = false
# figurePath = "/figures"


include("Ising.jl")

n = parse(Int64, ARGS[3]);
J = parse(Float64, ARGS[4]); n_iters = parse(Int64, ARGS[1]); h= parse(Float64, ARGS[5]);
d = parse(Int64, ARGS[2])
iter = 1:n_iters+1
if d == 2
    precompile(MCIsing,(Ising2DParam,))
    model = Ising2DParam(n, J, h, n_iters)
else
    precompile(MCIsing,(Ising3DParam,))
    model = Ising3DParam(n, J, h, n_iters)
end

M, E = @time MCIsing(model)

# Plot and save the energy trajectory

# plt.figure(1,tight_layout=true)
# plt.clf()
# plt.plot(iter,E,".k")
# plt.xlabel(r"$i$")
# plt.ylabel(r"$E$")
# plt.title(r"$E$  Trajectory")
# plt.savefig("E_vs_i.png")

# Plot and save the magnetization trajectory
# plt.figure(2,tight_layout=true)
# plt.clf()
# plt.plot(iter,M,".k")
# plt.xlabel(r"$i$")
# plt.ylabel(r"$M$")
# plt.title(r"$M$  Trajectory")
# plt.savefig("M_vs_i.png")