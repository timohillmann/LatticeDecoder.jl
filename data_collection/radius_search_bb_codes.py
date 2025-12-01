import sinter
import matplotlib.pyplot as plt
import numpy as np
import plotting_lib as pl
import os
pl.update_settings(True)


PATH = "~/Dropbox/Apps/Overleaf/low\ density\ lattice\ decoders\ for\ GKP/Figures/"

def plot_marker_style(color, marker="o", ls="solid", ms=4.5):
    return dict(
        markerfacecolor=pl.lighten_color(color, 0.5),
        markeredgecolor=color,
        markersize=ms,
        linestyle=ls,
        marker=marker,
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

def search_radius_plot(
    file,
    schedules=["parallel", "serial"],
    decoders=["nearest"],
    local_search=[True],
    local_search_lll=[True],
):

    dir = "/Users/timo/Documents/LatticeDecoder.jl/results/figures/" + "".join(file.split("/")[:-1])
    print(dir)
    os.makedirs(dir, exist_ok=True)

    data = sinter.stats_from_csv_files(
        f"/Users/timo/Documents/LatticeDecoder.jl/results/{file:s}.csv"
    )

    # _data = sinter.group_by(data, key=lambda stat: stat.json_metadata["decoder"])
    # _data = sinter.group_by(
    #     _data["lsd"], key=lambda stat: stat.json_metadata["schedule"]
    # )
    # _data = sinter.group_by(
    #     _data["serial"], key=lambda stat: stat.json_metadata["local_search"]
    # )
    # print(_data)

    for code in ["18_4_3_p2"]:
    # for code in ["30_4_5_p2", "48_4_7_p2"]:
        for dec in decoders:
            for ls in local_search:
                for ls_lll in local_search_lll:
                    for sch in schedules:
                        print("Schedule:", sch)
                        fig, ax = pl.create_fig()
                        sinter.plot_error_rate(
                            ax=ax,
                            stats=data,
                            # x_func = lambda stat: stat.json_metadata["d"],
                            x_func= lambda stat: stat.json_metadata["search_radius"],
                            filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                            and stat.json_metadata["local_search"] == ls
                            and stat.json_metadata["local_search_lll"] == ls_lll
                            and stat.json_metadata["schedule"] == sch
                            and stat.json_metadata["code"] == code
                            and np.sqrt(2 * np.pi) * stat.json_metadata["sigma"]
                                    in [0.3, 0.4, 0.5, 0.6],
                            group_func=lambda stat: fr"$\sigma$ = "
                            + f"{np.sqrt(2 * np.pi) * stat.json_metadata['sigma']}",
                            plot_args_func=lambda index, curve_id: plot_marker_style(
                                COLORS[index], MARKERS[index]
                            ),
                        )
                        ax.set_yscale("log")
                        ax.set_xlabel(r"Search Radius $\epsilon$")
                        ax.set_ylabel("Symbol Error Rate")
                        ax.set_title(
                            f"{code}, Local search: {ls}, Schedule: {sch}", fontsize=6
                        )
                        ax.legend(ncols=1)
                        ax.set_ylim(None, 1.)
                        ax.grid(True)
                        pl.tight_layout(fig)
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{code}_{dec}_{ls}_{ls_lll}_{sch}_radius.pdf"
                        )
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{code}_{dec}_{ls}_{ls_lll}_{sch}_radius.png"
                        )
                        plt.show()



def threshold_plot(
    file,
    schedules=["parallel"],
    decoders=["lsd"],
    local_search=[False],
    classical=False,
    local_search_lll=[True],
    radius=1.0
):

    dir = "/Users/timo/Documents/LatticeDecoder.jl/results/figures/" + "".join(file.split("/")[:-1])
    print(dir)
    os.makedirs(dir, exist_ok=True)

    data = sinter.stats_from_csv_files(
        f"/Users/timo/Documents/LatticeDecoder.jl/results/{file:s}.csv"
    )

    # _data = sinter.group_by(data, key=lambda stat: stat.json_metadata["decoder"])
    # _data = sinter.group_by(
    #     _data["lsd"], key=lambda stat: stat.json_metadata["schedule"]
    # )
    # _data = sinter.group_by(
    #     _data["serial"], key=lambda stat: stat.json_metadata["local_search"]
    # )
    # print(_data)

    for dec in decoders:
        for ls in local_search:
            for ls_lll in local_search_lll:
                for sch in schedules:
                    print("Schedule:", sch)
                    fig, ax = pl.create_fig()
                    sinter.plot_error_rate(
                        ax=ax,
                        stats=data,
                        # x_func = lambda stat: stat.json_metadata["d"],
                        x_func=lambda stat: np.sqrt(2 * np.pi)
                        * stat.json_metadata["sigma"],
                        filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                        and stat.json_metadata["local_search"] == ls
                        and stat.json_metadata["local_search_lll"] == ls_lll
                        and stat.json_metadata["schedule"] == sch
                        and stat.json_metadata["search_radius"] == (1.0 if dec == "nearest" else 0.0),
                        group_func=lambda stat: "d = " + f"{stat.json_metadata['d']}",
                        plot_args_func=lambda index, curve_id: plot_marker_style(
                            COLORS[index], MARKERS[index]
                        ),
                    )
                    ax.set_yscale("log")
                    ax.set_xlabel(r"Noise strength $\sigma$")
                    ax.set_ylabel("Error rate")
                    ax.set_title(
                        f"Decoder: {dec}, Local search: {ls}, LS LLL: {ls_lll} Schedule: {sch}", fontsize=6
                    )
                    ax.legend(ncols=2)
                    ax.set_ylim(None, 1.0)
                    ax.grid(True)
                    pl.tight_layout(fig)
                    fig.savefig(
                        f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{dec}_{ls}_{ls_lll}_{sch}.pdf"
                    )
                    fig.savefig(
                        f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{dec}_{ls}_{ls_lll}_{sch}.png"
                    )
                    plt.show()


def error_rate_scaling_plot(
    file,
    schedules=["parallel"],
    decoders=["lsd"],
    local_search=[False],
    classical=False,
    local_search_lll=[True],
):
    data = sinter.stats_from_csv_files(
        f"/Users/timo/Documents/LatticeDecoder.jl/results/{file:s}.csv"
    )

    dir = "/Users/timo/Documents/LatticeDecoder.jl/results/figures/" + "".join(file.split("/")[:-1])
    os.makedirs(dir, exist_ok=True)


    for dec in decoders:
        for ls in local_search:
            for ls_lll in local_search_lll:
                for sch in schedules:
                    fig, ax = pl.create_fig()
                    sinter.plot_error_rate(
                        ax=ax,
                        stats=data,
                        x_func=lambda stat: stat.json_metadata["d"],
                        # x_func =  lambda stat: np.sqrt(2 * np.pi) * stat.json_metadata["sigma"],
                        filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                        and stat.json_metadata["local_search"] == ls
                        and stat.json_metadata["schedule"] == sch
                        and stat.json_metadata["local_search_lll"] == ls_lll
                        # stat.json_metadata["d"] in [3, 5, 7, 9, 13, 17, 21],
                        and np.sqrt(2 * np.pi) * stat.json_metadata["sigma"]
                        in [0.3, 0.4, 0.5, 0.6],
                        group_func=lambda stat: rf"$\sigma$ = "
                        + f"{np.sqrt(2 * np.pi) * stat.json_metadata['sigma']}",
                        plot_args_func=lambda index, curve_id: plot_marker_style(
                            COLORS[index], MARKERS[index]
                        ),
                    )
                    ax.set_yscale("log")
                    ax.set_xlabel("Code Distance $d$")
                    ax.set_xticks(np.arange(3, 20, 2))
                    ax.set_ylabel("Error rate")
                    ax.set_title(
                        f"Decoder: {dec}, Local search: {ls}, LS LLL: {ls_lll} Schedule: {sch}", fontsize=6
                    )
                    ax.legend()
                    ax.set_ylim(None, 0.75)
                    ax.grid(True)
                    pl.tight_layout(fig)
                    fig.savefig(
                        f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_scaling_{dec}_{ls}_{ls_lll}_{sch}.pdf"
                    )
                    fig.savefig(
                        f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_scaling_{dec}_{ls}_{ls_lll}_{sch}.png"
                    )
                    plt.show()


# path = "quantum_codes/bb_codes_results_lx"
# path = "classical_codes/balanced_last_reduced_rep_code_ls"
paths =[
    # "quantum_rep_code/standard_reduced_check_matrix_ls",
    # "quantum_rep_code/balanced_first_reduced_check_matrix_ls",
    # "quantum_rep_code/balanced_last_reduced_check_matrix_ls",
    # "quantum_codes/small_bb_codes_balanced_radius",
    "quantum_codes/small_bb_codes_balanced_radius",
    # "quantum_codes/small_bb_codes_unbalanced_radius",
    ]
schedules = ["serial"]
decoders = ["nearest"]
local_search = [True]
local_search_lll = [True]

# TODO: I need to add another parameter for reduced / non-reduced.

for path in paths:
    search_radius_plot(
        path,
        schedules=schedules,
        decoders=decoders,
    )
    # threshold_plot(
    #     path,
    #     local_search=local_search,
    #     decoders=decoders,
    #     schedules=schedules,
    #     local_search_lll=local_search_lll,
    # )
    # error_rate_scaling_plot(
    #     path,
    #     local_search=local_search,
    #     decoders=decoders,
    #     schedules=schedules,
    #     local_search_lll=local_search_lll,
    # )
