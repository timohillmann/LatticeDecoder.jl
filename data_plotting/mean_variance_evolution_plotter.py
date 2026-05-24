from json import load
import matplotlib.pyplot as plt
import numpy as np
import plotting_lib as pl
import os
pl.update_settings(True)

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='C0', lw=1.5),  # thick blue line
    Line2D([0], [0], color='C1', lw=1.5)  # dashed orange line
]

# def load_code(code_name):
#     path = f"/Users/timo/Documents/LatticeDecoder.jl/data/classical_ldlc_matrices/{code_name}_H.npy"    
#     return np.load(path)


# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/timeseries/ldlc_n128_d5_parallel_nearest.npz"


SAVE_PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/timeseries/plots/"

def make_plots(path, alpha=0.8, lw=0.75):
    data = np.load(path)
    code = path.split("/")[-1].strip(".npz")
    var = data["var"]
    MASK = data["mask"]
    NOT_MASK = data["NOT_MASK"]
    means = data["means"]




    labels = [r"$\lvert h_i \rvert = 1$", r"$\lvert h_i \rvert < 1$"]
    # VAR PLOT
    fig, ax = pl.create_fig()
    curves0 = ax.plot(var[MASK-1, :].T, color="C0", lw=lw, alpha=alpha, label=r"$\lvert h_i \rvert = 1$")
    # ax.plot(np.mean(var[MASK-1, :].T, axis=1), color="k")
    curves1 = ax.plot(var[NOT_MASK-1, :].T, color="C1", lw=lw, alpha=alpha, label=r"$\lvert h_i \rvert < 1$")
    # ax.plot(np.mean(var[NOT_MASK-1, :].T, axis=1), color="k")
    ax.legend(handles=custom_lines, labels=labels,  borderaxespad=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $s$")
    ax.set_ylabel("Variable Node Variance")

    # code_name = "_".join(code.split("_")[:3])
    # H = load_code(code_name)
    # dec = np.round(H * var[:, -1])
    # print(np.max(np.abs(dec)))

    fig.savefig(SAVE_PATH + f"{code}_var.png")

    # MEAN PLOT
    fig, ax = pl.create_fig()
    curves0 = ax.plot(means[MASK-1, :].T, color="C0", lw=lw, alpha=alpha, label=r"$\lvert h_i \rvert = 1$")
    # ax.plot(np.mean(var[MASK-1, :].T, axis=1), color="k")
    curves1 = ax.plot(means[NOT_MASK-1, :].T, color="C1", lw=lw, alpha=alpha, label=r"$\lvert h_i \rvert < 1$")
    # ax.plot(np.mean(var[NOT_MASK-1, :].T, axis=1), color="k")
    ax.set_xlabel("Iteration $s$")
    ax.set_ylabel("Variable Node Mean")
    ax.legend(handles=custom_lines, labels=labels, borderaxespad=0.75)
    fig.savefig(SAVE_PATH + f"{code}_mean.png")
    plt.show()

if __name__ == "__main__":
    codes = ["ldlc_n128_d5", "ldlc_n256_d5", "ldlc_n512_d5",]
    codes = ["ldlc_n768_d7"]
    codes = ["rep_code_5_first", "rep_code_5_last", "rep_code_5_standard"]
    codes = [f"surface_code_{d}_{balanced}" for balanced in ["true", "false"] for d in [3, 5, 7]]
    codes = [f"{code}_{balanced}" for code in ["30_4_5_p2", "48_4_7_p2"] for balanced in ["true", "false"]]
    codes = ["toric_3D_3_true", "toric_3D_3_false"]
    for schedule in ["serial", "parallel"]:
        for code in codes:
            for decoder in ["nearest", "lsd"]:
                path = f"/Users/timo/Documents/LatticeDecoder.jl/data/timeseries/{code}_{schedule}_{decoder}.npz"
                print(code, schedule, decoder)
                make_plots(path=path)