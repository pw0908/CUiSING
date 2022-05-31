using Printf

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

open("output/julia_cpu_output.dat", "w") do f
    for i = iter
        if mod(i,10) == 0
            println(f, i," ",@sprintf("%.5f",E[i])," ", @sprintf("%.5f",M[i]))
        end
    end
end