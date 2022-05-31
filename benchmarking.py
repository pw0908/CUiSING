# Import modules
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
import os
import re
import numpy as np
import csv
from progressbar import ProgressBar
pbar = ProgressBar()
import timeit
import sys
import os
sys.path.append("python/")

# Set figure parameters
WIDTH = 1.5 * 8.3 / 2.54
DOUBLE_WIDTH = 1.5 * 17.1 / 2.54
DPI = 350
format='png'
matplotlib.rcParams.update(
    {
        'axes.labelsize': 14,
        'axes.xmargin': 0,
        'axes.ymargin': .1,
        'lines.markersize': 3,
        'figure.dpi': DPI,
        'figure.autolayout': True,
        'figure.figsize': (WIDTH, 3 * WIDTH / 4),
        'figure.facecolor': 'white',
        'font.size': 12,
        'grid.color': '0',
        'grid.linestyle': '-',
        'legend.edgecolor': '1',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': "DeJavu Serif",
        'font.serif': ["Computer Modern Roman"],
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'text.usetex': False
    }
)


n = np.logspace(2,2,1).astype(int)
n_iters = int(1e3)
J = 0.1
h = 0
d = int(3)

pc = "Sam_add"
delim = "_"

figure_root = "benchmarking/"+str(d)+"d/figures/"
data_root = "benchmarking/"+str(d)+"d/data/"

figure_name = pc+delim+str(n_iters)+delim+str(J)+delim+str(h)+delim+str(d)+"."+format
data_name = pc+delim+str(n_iters)+delim+str(J)+delim+str(h)+delim+str(d)+".dat"

if os.path.exists(data_root+data_name):
    os.remove(data_root+data_name)

t_cpp = np.zeros(len(n))
t_julia = np.zeros(len(n))
t_python = np.zeros(len(n))

setup = '''from Ising import Ising2D,Ising3D'''

for i in pbar(range(len(n))):
    process_cpp = subprocess.run(["./cpp/Ising",str(n_iters), str(d), str(n[i]), str(J), str(h)], capture_output=True)
    t_cpp[i] = re.search('Program Time : (.*) seconds', str(process_cpp.stdout)).group(1)
    process_julia = subprocess.run(["julia julia/benchmarking.jl "+str(int(n_iters))+" "+str(d)+" "+str(int(n[i]))+" "+str(J)+" "+str(h)], capture_output=True, shell = True)
    t_julia[i] = re.search('(.*) seconds', str(process_julia.stdout)).group(1).split("b'")[-1].split()[-1]
    if d == 2:
        stmt = '''model = Ising2D(n[i],J,h,n_iters)
M,E = model.run()'''
    elif d == 3:
        stmt = '''model = Ising3D(n[i],J,h,n_iters)
M,E = model.run()'''
    t_python[i] = timeit.timeit(setup=setup,stmt=stmt,number=1,globals=globals())
    with open(data_root+data_name,'a') as f:
        f.write(str(n[i])+" "+str(t_cpp[i])+" "+str(t_julia[i])+" "+str(t_python[i])+"\n")

plt.figure(1,tight_layout=True)
plt.loglog(n,t_cpp,'.b', label = "C++")
plt.loglog(n,t_julia,'.g', label = "Julia")
plt.loglog(n,t_python,'.r',label="Python")
plt.xlabel(r'$n$')
plt.ylabel(r'$t / $s')
plt.title(rf'{d}D Benchmarks')
plt.legend(fontsize=12, handlelength=1.25, columnspacing=1, labelspacing=0.25)
plt.savefig(figure_root+figure_name)

with open(data_root+data_name,'w',newline='') as f:
    linewriter = csv.writer(f,delimiter=' ', quoting=csv.QUOTE_MINIMAL,dialect='unix')
    for n_, tc, tj, tp in zip(n, t_cpp, t_julia, t_python):
                contents = [n_,tc,tj,tp]
                linewriter.writerow(contents)