import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

textwidth = 7.10000594991
columnwidth = 3.35224200913

fig, ax = plt.subplots(1, figsize=(textwidth, 3))

x = np.linspace(0, 7*np.pi, 5000)
y = np.cos(x)
y2 = 0.25*np.cos(x + np.pi) - 1.25
y3 = 2.25*np.cos(x) - 1.25

ax.plot(x, y, c=tb_c[-1])

ax.plot(x, y2, c=tb_c[4], zorder=4)
ax.scatter(3.*np.pi, -1, c=tb_c[4], zorder=4)

ax.plot(x, y3, c=tb_c[6], zorder=5)
ax.scatter(4.*np.pi, 1, c=tb_c[6], zorder=5)

ax.plot(x, np.full(len(x), 0), c=tb_c[-1])
ax.plot(x, np.full(len(x), -1.25), c=tb_c[-1], ls='dashed')
#ax.axis('off')

ax.set_xlabel(r'$\text{orbital angle}$')
ax.set_ylabel(r'$z$')

ax.set_xticklabels([])
ax.set_yticklabels([])

fig.tight_layout()
fig.savefig('cartoon.pdf', bbox_inches='tight', pad_inches=0)
