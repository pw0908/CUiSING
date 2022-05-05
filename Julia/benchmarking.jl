using Test

include("Ising.jl")
n = 100;
J = 0.5; n_iters = 10^4; sample_freq = 20; h=0;

@time MCIsing(n, J, h, n_iters)