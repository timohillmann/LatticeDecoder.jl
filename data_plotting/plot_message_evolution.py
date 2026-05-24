from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import plotting_lib as pl
import os
pl.update_settings(True)


def plot_marker_style(color, marker="o", ls="solid", ms=4.5):
    return dict(
        markerfacecolor=pl.lighten_color(color, 0.5),
        markeredgecolor=color,
        markersize=ms,
        linestyle=ls,
        marker=marker,
        color=color,
    )


COLORS = pl.colors_rsb
FONTSIZE = 9
MARKERS = [
    "o",
    "s",
    "d",
    "p",
    "X",
    "^",
    "v",
    "<",
    ">",
    "h",
    "H",
    "D",
    "P",
    "x",
    "|",
    "_",
]


def gaussian(x,m, var):
    return np.exp(-(x - m)**2 / (2 * var))

nv = 3
n_iter = 6
fig, ax = pl.create_fig(ncols=nv, nrows=n_iter, sharex=True, sharey=True, single_col=True, height=5.25)

code = "rep_code_3_last"
schedule = "parallel"
decoder = "lsd"

#  {code}_{schedule}_{decoder}
path = f"/Users/timo/Documents/LatticeDecoder.jl/data/timeseries/trace_timeseries/rep_code_3_lsd_balanced_last_v3.npz"

data = np.load(path)
var = data["vars"].T
means = data["means"].T
decoded = data["decoded"]
X = np.linspace(-np.sqrt(2) - 0.2, 0.2 + np.sqrt(2), 5000)




for vn in range(nv):
    for iter in range(n_iter):
        ax[iter, vn].plot(X, gaussian(X, means[vn, 0], var[vn, 0]), c="grey", lw=0.95, ls="dashed", zorder=-3)
        ax[iter, vn].plot(X, gaussian(X, means[vn, iter], var[vn, iter]), c="C1", zorder=-1)
        ax[iter, vn].fill_between(X, gaussian(X, means[vn, iter], var[vn, iter]), 0, color=pl.lighten_color("C1", 0.1), zorder=-2)

for axis in ax.flatten():
    # axis.grid(True, which="both")
    # axis.set_axisbelow(True)
    # axis.set_aspect("equal")
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(axis='y', left=False, labelleft=False, right=False, which="both")
    axis.tick_params(axis='x', top=False, which="both", pad=3.5)
    axis.set_ylim(0, 1.1)
    axis.set_xlim(-np.sqrt(2) - 0.2, np.sqrt(2) + 0.2)
    axis.set_xticks([-np.sqrt(2), -np.sqrt(2)/2, 0.0, np.sqrt(2)/2, np.sqrt(2)], minor=True)
    axis.set_xticks([-np.sqrt(2), 0.0, np.sqrt(2)])
    axis.set_xticklabels([r"$-\sqrt{2}$", r"$0$",r"$\sqrt{2}$"], fontsize=7)
    # axis.set_xticks([r"$-\sqrt{2}$", r"$-\sqrt{2}/2$", r"$0", r"$\sqrt{2}/2$",r"$\sqrt{2}$"])

    # axis.set_xticks([-1.0, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.0], minor=True)
    # axis.set_xticks([-1.0, 0.,1.0], minor=False)
    # axis.set_xticklabels([-1,  0., 1])
    # axis.set_yticks([0., 0.25, 0.5, 0.75, 1.0], minor=True)
    # axis.set_yticks([0., 0.25, 0.5, 0.75, 1.0], minor=False)


for idx, axis in enumerate(ax[0, :]):
    axis.set_title(f"Variable Node {idx+1}")

for idx, axis in enumerate(ax[-1, :]):
    axis.set_xlabel(rf"Coordinate $x_{{{idx + 1}}}$")



for idx, axis in enumerate(ax[:, 0]):
    axis.set_ylabel(f"Iteration {idx}")

for idx, axis in enumerate(ax[-1, :]):
    axis.axvline(decoded[idx], ls="dashed", ymax = gaussian(decoded[idx], means[idx, n_iter-1], var[idx, n_iter-1]) / 1.1, zorder=-2)

pl.tight_layout()
fig.subplots_adjust(wspace=0.10, hspace=0.05)
pl.add_label(ax[0, 0], text="a", x0=0.15, y0=1.25)
fig.savefig(f"/Users/timo/Dropbox/Apps/Overleaf/paper_quantum_ldlc/figures/rep_code_message_evolution_v3.pdf")

# for axis in ax.flatten():