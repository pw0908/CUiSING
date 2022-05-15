# Import modules
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
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

# Set system parameters
n_iters = 100 # Number of MC iterations
d = 2           # Number of dimensions
n = 1000         # Number of cells per dimension
J = .1          # Interaction strength
h = 0.0         # Magnetic field strength

# Setup output file structure
outputFile = "output/output.dat"
delim = '_'
figureRoot = 'figures/'
identifier = 'Sys_' + str(n_iters) + delim + str(d) + delim + str(n) + delim + str(J) + delim + str(h)
figurePath = figureRoot+identifier

if not (os.path.exists(figureRoot)):
    os.mkdir(figureRoot)
    os.mkdir(figurePath)
elif not (os.path.exists(figurePath)):
    os.mkdir(figurePath)

# Run ising process, with input arguments
subprocess.run(["./Ising",str(n_iters), str(d), str(n), str(J), str(h)])

# Read in the output data
iter = []
E = []
M = []
with open(outputFile,'rb') as f:
    for line in f:
        line = line.split()
        iter += [int(line[0])]
        E += [float(line[1])]
        M += [float(line[2])]

# Plot and save the energy trajectory
plt.figure(1,tight_layout=True)
plt.plot(iter,E,'.k')
plt.xlabel(r'$i$')
plt.ylabel(r'$E$')
plt.title(r'$E$  Trajectory')
plt.savefig(figurePath+'/E_vs_i.png')

# Plot and save the magnetization trajectory
plt.figure(2,tight_layout=True)
plt.plot(iter,M,'.k')
plt.xlabel(r'$i$')
plt.ylabel(r'$M$')
plt.title(r'$M$  Trajectory')
plt.savefig(figurePath+'/M_vs_i.png')




