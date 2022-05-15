# Import modules
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
import os
import re
import numpy as np


n = np.logspace(1,3,50)
n_iters = 1e3
J = 0.5
h = 0
d = 2

t_cpp = np.zeros(len(n))
t_julia = np.zeros(len(n))

for i in range(len(n)):
    process_cpp = subprocess.run(["./cpp/Ising",str(n_iters), str(d), str(n[i]), str(J), str(h)], capture_output=True)
    t_cpp[i] = re.search('Program Time : (.*) seconds', str(process_cpp.stdout)).group(1)
    process_julia = subprocess.run(["julia Julia/benchmarking.jl "+str(int(n_iters))+" "+str(d)+" "+str(int(n[i]))+" "+str(J)+" "+str(h)], capture_output=True, shell = True)
    t_julia[i] = re.search(' (.*) seconds', str(process_julia.stdout)).group(1)

plt.figure(1,tight_layout=True)
plt.loglog(n,t_cpp,'.b', label = "C++")
plt.loglog(n,t_julia,'.g', label = "Julia")
plt.xlabel(r'$n$')
plt.ylabel(r'$t / $s')
plt.title(r'2D Benchmarks')
plt.legend()
plt.savefig('2D_benchmark.png')