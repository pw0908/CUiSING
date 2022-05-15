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


n = np.logspace(1,3,50).astype(int)
n_iters = int(1e3)
J = 0.1
h = 0
d = int(2)

pc = "Sam"
delim = "_"

figure_root = "benchmarking/"+str(d)+"d/figures/"
data_root = "benchmarking/"+str(d)+"d/data/"

figure_name = pc+delim+str(n_iters)+delim+str(J)+delim+str(h)+delim+str(d)+"."+format
data_name = pc+delim+str(n_iters)+delim+str(J)+delim+str(h)+delim+str(d)+".dat"

t_cpp = np.zeros(len(n))
t_julia = np.zeros(len(n))

for i in pbar(range(len(n))):
    process_cpp = subprocess.run(["./cpp/Ising",str(n_iters), str(d), str(n[i]), str(J), str(h)], capture_output=True)
    t_cpp[i] = re.search('Program Time : (.*) seconds', str(process_cpp.stdout)).group(1)
    process_julia = subprocess.run(["julia Julia/benchmarking.jl "+str(int(n_iters))+" "+str(d)+" "+str(int(n[i]))+" "+str(J)+" "+str(h)], capture_output=True, shell = True)
    t_julia[i] = re.search(' (.*) seconds', str(process_julia.stdout)).group(1)

plt.figure(1,tight_layout=True)
plt.loglog(n,t_cpp,'.b', label = "C++")
plt.loglog(n,t_julia,'.g', label = "Julia")
plt.xlabel(r'$n$')
plt.ylabel(r'$t / $s')
plt.title(rf'{d}D Benchmarks')
plt.legend(fontsize=12, handlelength=1.25, columnspacing=1, labelspacing=0.25)
plt.savefig(figure_root+figure_name)

with open(data_root+data_name,'w',newline='') as f:
    linewriter = csv.writer(f,delimiter=' ', quoting=csv.QUOTE_MINIMAL,dialect='unix')
    for n_, tc, tj in zip(n, t_cpp, t_julia):
                contents = [n_,tc,tj]
                linewriter.writerow(contents)