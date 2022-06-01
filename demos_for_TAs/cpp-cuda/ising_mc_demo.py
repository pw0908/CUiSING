# Import modules
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
import os
import numpy as np
import csv
import sys
import os

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

n_iters = int(1e4)
d = int(2)
n = 300
J = 0.5
h = 0.0

subprocess.run(["./../../cpp-cuda/Ising",str(n_iters), str(d), str(n), str(J), str(h)], capture_output=False)

file = "output/cpp_gpu_output.dat"
iters = []
Es = []
Ms = []
with open(file,'rb') as f:
    for line in f:
        iters += [int(line.split()[0])]
        Es += [float(line.split()[1])]
        Ms += [float(line.split()[2])]

plt.figure(1)
plt.plot(iters,Es,'-k',label=r"$E$")
plt.plot(iters,Ms,'-r',label=r"$M$")
plt.xlabel("Iters")
plt.title("Trajectories")
plt.savefig("figures/trajectories.png",bbox_inches='tight')


cmap = matplotlib.colors.ListedColormap(["black","white"], name='Ising', N=None)
file = "output/cpp_gpu_lattice.dat"
A = np.loadtxt(file,dtype=int)
plt.figure(2,tight_layout=True)
plt.title("Lattice")
plt.imshow(A,cmap=cmap)
plt.colorbar(ticks=[-1,1])
plt.savefig("figures/lattice.png",bbox_inches='tight')