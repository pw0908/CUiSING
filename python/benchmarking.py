import matplotlib
import timeit
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n_iter")
parser.add_argument("d")
parser.add_argument("n")
parser.add_argument("J")
parser.add_argument("h")
args = parser.parse_args()
n_iter = int(args.n_iter)
d = int(args.d)
n = int(args.n)
J = float(args.J)
h = float(args.h)

setup = '''
from Ising import Ising2D
'''

stmt = '''
model = Ising2D(n,J,h,n_iter)
M,E = model.run()
'''
print(f"Program time : {timeit.timeit(setup=setup,stmt=stmt,number=1,globals=globals())} seconds")

# t=time.time()
# model = Ising2D(n,J,h,n_iter)
# M1,E1 = model.run()
# print(f"Program time, slow: {time.time()-t} seconds")
# print(f"M, E = {M1[-1]}, {E1[-1]}")

# figurePath = "figures"
# iter = range(n_iter)
# # Plot and save the energy trajectory
# plt.figure(1,tight_layout=True)
# plt.plot(iter,E,'.k')
# plt.xlabel(r'$i$')
# plt.ylabel(r'$E$')
# plt.title(r'$E$  Trajectory')
# plt.savefig(figurePath+'/E_vs_i.png')

# # Plot and save the magnetization trajectory
# plt.figure(2,tight_layout=True)
# plt.plot(iter,M,'.k')
# plt.xlabel(r'$i$')
# plt.ylabel(r'$M$')
# plt.title(r'$M$  Trajectory')
# plt.savefig(figurePath+'/M_vs_i.png')