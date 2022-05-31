# Import modules
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
import os
import numpy as np
import csv
from progressbar import ProgressBar
import shutil
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

n = 200
n_iters = int(1e6)
J_vec = np.concatenate((np.linspace(0.01,0.4,10,endpoint=False),np.linspace(0.4,0.52,20,endpoint=False),np.linspace(0.52,1.0,15)))
# J_vec = np.linspace(0.4,0.52,20)
h = 0.0
d = int(2)

if os.path.exists("output/"):
    shutil.rmtree("output/")
    os.mkdir("output/")

for J in pbar(J_vec):
    process = subprocess.run(["./../cpp-cuda/Ising",str(n_iters), str(d), str(n), str(J), str(h)], capture_output=False, 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.rename('output/cpp_gpu_output.dat','output/cpp_gpu_output_'+str(round(J,4))+'.dat')
