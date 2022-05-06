import os
import numpy as np
import subprocess
import matplotlib
from matplotlib import pyplot as plt

WIDTH = 1.5 * 8.3 / 2.54
DOUBLE_WIDTH = 1.5 * 17.1 / 2.54
DPI = 500
format='eps'
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
        'grid.linestyle': (0, (1, 5)),
        'legend.edgecolor': '1',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': "DeJavu Serif",
        'font.serif': ["Computer Modern Roman"],
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}'
    }
)

n_iters = 10000
d = 2
n = 100
J = .5
h = 0.0
outputFile = "output/output.dat"

subprocess.run(["./Ising",str(n_iters), str(d), str(n), str(J), str(h)])

iter = []
E = []
M = []

with open(outputFile,'rb') as f:
    for line in f:
        line = line.split()
        iter += [int(line[0])]
        E += [float(line[1])]
        M += [float(line[2])]

plt.figure(1)

plt.plot(iter,E,'.k')
plt.xlabel(r'')


