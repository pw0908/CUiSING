import matplotlib
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
from Ising import Ising2D

model = Ising2D(100,0.5,0.,1000)

M,E = model.run()

figurePath = "figures"
iter = range(1000)
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