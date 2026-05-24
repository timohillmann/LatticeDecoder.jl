from typing import Union
import sinter
import matplotlib.pyplot as plt
import numpy as np
import plotting_lib as pl
import os
import mpmath as mp
import scipy.interpolate as scinp
pl.update_settings(True)



def interpolate_data(x, y, ratio=15):
    X = np.linspace(x.min(), x.max(), len(x) * ratio)
    _Y = scinp.CubicSpline(x, y)
    return X, _Y(X)




def prob_erf_n(sig, n):
    t1 = mp.erf((1 - 4 * n) * mp.sqrt(mp.pi / 2) / (2 * sig))
    t2 = mp.erf((1 + 4 * n) * mp.sqrt(mp.pi / 2) / (2 * sig))
    return 1 / 2 * (t1 + t2)

def prob_erf(sig):
    out = mp.mpf(0)
    npeaks = 100 # int(3 / sig)
    for n in range(-npeaks, npeaks+1):
        out += prob_erf_n(sig, n)
    return float(out)


prob_erf_n = np.vectorize(prob_erf_n)
prob_erf = np.vectorize(prob_erf)

def gkp_err_prob(sigmas):
    return 1 - prob_erf(sigmas)**4


def plot_marker_style(color, marker="o", ls="solid", ms=4.5):
    return dict(
        markerfacecolor=pl.lighten_color(color, 0.5),
        markeredgecolor=color,
        markersize=ms,
        linestyle=ls,
        marker=marker,
        color=color,
    )




COLORS = pl.colors_rsb * 103
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
] * 100

def search_radius_plot(
    file,
    schedules=["parallel", "serial"],
    decoders=["nearest"],
    local_search=[True],
    local_search_lll=[True],
    ds =  [3, 5, 7, 9, 11, 13, 15],
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

    for d in ds:
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
                            and stat.json_metadata["local_search_lll"] == ls_lll if ls else False
                            and stat.json_metadata["schedule"] == sch
                            and stat.json_metadata["d"] == d
                            and np.sqrt(2 * np.pi) * stat.json_metadata["sigma"]
                                    in [0.3, 0.4, 0.5, 0.6],
                            group_func=lambda stat: rf"$\sigma$ = "
                            + f"{np.sqrt(2 * np.pi) * stat.json_metadata['sigma']}",
                            plot_args_func=lambda index, curve_id: plot_marker_style(
                                COLORS[index], MARKERS[index]
                            ),
                        )
                        ax.set_yscale("log")
                        ax.set_xlabel(r"Search Radius $\epsilon$")
                        ax.set_ylabel("Symbol Error Rate")
                        ax.set_title(
                            f"d = {d}, Local search: {ls}, Schedule: {sch}", fontsize=6
                        )
                        ax.legend(ncols=1)
                        ax.set_ylim(None, 0.75)
                        ax.grid(True)
                        pl.tight_layout(fig)
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{d}_{dec}_{ls}_{ls_lll}_{sch}_radius.pdf"
                        )
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{d}_{dec}_{ls}_{ls_lll}_{sch}_radius.png"
                        )
                        plt.show()



def threshold_plot(
    file,
    schedules=["parallel"],
    decoders=["lsd"],
    local_search=[False],
    local_search_lll=[True],
    balance_weights=[False],
    sphere_decoding=True,
    k=2,
    printdata=False,
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
    
    if printdata:
        for stat in data:
            print(stat.to_csv_line())

    for dec in decoders:
        for ls in local_search:
            for bw in balance_weights:
                for ls_lll in local_search_lll:
                    ls_lll = ls_lll if ls else False
                    sphere_decoding = sphere_decoding if ls else False
                    for sch in schedules:
                        fig, ax = pl.create_fig()
                        sinter.plot_error_rate(
                            ax=ax,
                            stats=data,
                            x_func=lambda stat: np.sqrt(2 * np.pi)
                            * stat.json_metadata["sigma"],
                            filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                            and stat.json_metadata["local_search"] == ls
                            and stat.json_metadata["local_search_lll"] == ls_lll
                            and stat.json_metadata["schedule"] == sch
                            and stat.json_metadata["d"] % 2 == 1
                            and stat.json_metadata["balance_weights"] == bw
                            and stat.json_metadata["sphere_decoding"] == sphere_decoding,
                            group_func=lambda stat:  {'label': f"d = {stat.json_metadata['d']}", 
                         'sort': stat.json_metadata['d']},
                            plot_args_func=lambda index, curve_id: plot_marker_style(
                            COLORS[index], MARKERS[index]),
                        )
                        ax.set_yscale("log")
                        ax.set_xlabel(r"Noise strength $\sigma$")
                        ax.set_ylabel("Logical error rate")
                        ax.set_title(
                            f"Decoder: {dec}, Local search: {ls}, LS LLL: {ls_lll} Schedule: {sch}", fontsize=6
                        )
                        ax.legend(ncols=1, fontsize=9)
                        ax.set_ylim(None, 1.)

                        sigma_min, sigma_max = ax.get_xlim()                            
                        _sigmas_ = np.linspace(sigma_min, sigma_max, 512)
                        ax.fill_between(_sigmas_, gkp_err_prob(_sigmas_), 1, color=pl.lighten_color("k", 0.1))
                        ax.plot(_sigmas_, gkp_err_prob(_sigmas_), ls="dashed", color=pl.lighten_color("k", 0.5))


                        ax.grid(True)
                        pl.tight_layout(fig)
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{dec}_{ls}_{ls_lll}_{sch}.pdf"
                        )
                        fig.savefig(
                            f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{dec}_{ls}_{ls_lll}_{sch}.png"
                        )
                        plt.show()


def plot_paper(
    file,
    schedules=["parallel"],
    decoders=["lsd"],
    local_search=[False],
    local_search_lll=[True],
    balance_weights=[False],
    sphere_decoding=True,
    k=2,
    printdata=False,
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
    
    if printdata:
        for stat in data:
            print(stat.to_csv_line())
    labels = {"30_4_5_p2": r"$\llbracket 30, 4, 5 \rrbracket$", "48_4_7_p2": r"$\llbracket 48, 4, 7 \rrbracket$"}
    for dec in decoders:
        for ls in local_search:
            for bw in balance_weights:
                for ls_lll in local_search_lll:
                    ls_lll = ls_lll if ls else False
                    sphere_decoding = sphere_decoding if ls else False
                    for sch in schedules:
                        fig, ax = pl.create_fig()
                        sinter.plot_error_rate(
                            ax=ax,
                            stats=data,
                            x_func=lambda stat: np.sqrt(2 * np.pi)
                            * stat.json_metadata["sigma"],
                            filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                            and stat.json_metadata["local_search"] == ls
                            and stat.json_metadata["local_search_lll"] == ls_lll
                            and stat.json_metadata["schedule"] == sch
                            and stat.json_metadata["balance_weights"] == bw
                            and stat.json_metadata["sphere_decoding"] == sphere_decoding
                            and stat.json_metadata["sigma"] * np.sqrt(2*np.pi) > 0.3,
                            group_func=lambda stat: labels[stat.json_metadata['code_name']],
                            plot_args_func=lambda index, curve_id: plot_marker_style(
                            COLORS[index], MARKERS[index]),
                        )
                        ax.set_yscale("log")
                        ax.set_xlabel(r"Noise strength $\sigma$")
                        ax.set_ylabel("Logical $X$ error rate")
                        # ax.set_title(
                        #     f"Decoder: {dec}, Local search: {ls}, LS LLL: {ls_lll} Schedule: {sch}", fontsize=6
                        # )
                        ax.legend(fontsize=9, loc="lower right")
                        ax.set_ylim(None, 1.)

                        sigma_min, sigma_max = ax.get_xlim()                            
                        _sigmas_ = np.linspace(sigma_min, sigma_max, 512)
                        ax.fill_between(_sigmas_, gkp_err_prob(_sigmas_), 1, color=pl.lighten_color("k", 0.1))
                        ax.plot(_sigmas_, gkp_err_prob(_sigmas_), ls="dashed", color=pl.lighten_color("k", 0.5))


                        ax.grid(True)
                        pl.tight_layout(fig)
                        fig.savefig(
                            f"/Users/timo/Dropbox/Apps/Overleaf/paper_quantum_ldlc/figures/bb_codes_decoder_performance.pdf"
                        )
                        # fig.savefig(
                        #     f"/Users/timo/Documents/LatticeDecoder.jl/results/figures/{file}_{dec}_{ls}_{ls_lll}_{sch}.png"
                        # )
                        plt.show()


def code_fam_plot(
    file,
    schedules=["parallel"],
    decoders=["lsd"],
    local_search=[False],
    local_search_lll=[True],
    balance_weights=[False],
    sphere_decoding=True,
    k=2,
    printdata=False,
):

    dir = "/Users/timo/Documents/LatticeDecoder.jl/results/figures/" + "".join(file.split("/")[:-1])
    print(dir)
    os.makedirs(dir, exist_ok=True)

    data = sinter.stats_from_csv_files(
        f"/Users/timo/Documents/LatticeDecoder.jl/results/{file:s}.csv"
    )

    _data = sinter.group_by(data, key=lambda stat: stat.json_metadata["code_name"].split("_")[0])

    # _data = sinter.group_by(
    #     _data["lsd"], key=lambda stat: stat.json_metadata["schedule"]
    # )
    # _data = sinter.group_by(
    #     _data["serial"], key=lambda stat: stat.json_metadata["local_search"]
    # )
    # print(_data)
    
    if printdata:
        for stat in data:
            print(stat.to_csv_line())

    for key, _data_ in _data.items():
        for dec in decoders:
            for ls in local_search:
                for bw in balance_weights:
                    for ls_lll in local_search_lll:
                        ls_lll = ls_lll if ls else False
                        sphere_decoding = sphere_decoding if ls else False
                        for sch in schedules:
                            fig, ax = pl.create_fig()
                            sinter.plot_error_rate(
                                ax=ax,
                                stats=_data_,
                                x_func=lambda stat: np.sqrt(2 * np.pi)
                                * stat.json_metadata["sigma"],
                                filter_func=lambda stat: stat.json_metadata["decoder"] == dec
                                and stat.json_metadata["local_search"] == ls
                                and stat.json_metadata["local_search_lll"] == ls_lll
                                and stat.json_metadata["schedule"] == sch
                                and stat.json_metadata["balance_weights"] == bw
                                and stat.json_metadata["sphere_decoding"] == sphere_decoding,
                                group_func=lambda stat: f"{stat.json_metadata['code_name']}",
                                plot_args_func=lambda index, curve_id: plot_marker_style(
                                COLORS[index], MARKERS[index]),
                            )
                            ax.set_yscale("log")
                            ax.set_xlabel(r"Noise strength $\sigma$")
                            ax.set_ylabel("Logical error rate")
                            ax.set_title(
                                f"Decoder: {dec}, Local search: {ls}, LS LLL: {ls_lll} Schedule: {sch}", fontsize=6
                            )
                            # ax.legend(ncols=3, fontsize=6)
                            ax.set_ylim(None, 1.)

                            sigma_min, sigma_max = ax.get_xlim()                            
                            _sigmas_ = np.linspace(sigma_min, sigma_max, 512)
                            ax.plot(_sigmas_, gkp_err_prob(_sigmas_), ls="dashed", c="grey")
                            

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
    local_search_lll=[True],):
    data = sinter.stats_from_csv_files(
        f"/Users/timo/Documents/LatticeDecoder.jl/results/{file:s}.csv"
    )

    dir = "/Users/timo/Documents/LatticeDecoder.jl/results/figures/" + "".join(file.split("/")[:-1])
    os.makedirs(dir, exist_ok=True)


    for dec in decoders:
        for ls in local_search:
            for ls_lll in local_search_lll:
                ls_lll = ls_lll if ls else False
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
                        and stat.json_metadata["sphere_decoding"] == False
                        # stat.json_metadata["d"] in [3, 5, 7, 9, 13, 17, 21],
                        and np.sqrt(2 * np.pi) * stat.json_metadata["sigma"]
                        in [0.3, 0.4, 0.5, 0.6],
                        group_func=lambda stat: rf"$\sigma$ = "
                        + f"{np.sqrt(2 * np.pi) * stat.json_metadata['sigma']}",
                        plot_args_func=lambda index, group_key, group_stats: plot_marker_style(
                            COLORS[group_key], MARKERS[group_key]
                        ),
                    )
                    ax.set_yscale("log")
                    ax.set_xlabel("Code Distance $d$")
                    ax.set_xticks(np.arange(3, 20, 2))
                    ax.set_ylabel("Logical error rate")
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
    # "quantum_codes/bivariate_bicycle_w5",
    # "quantum_codes/bivariate_bicycle_w5b",
    # "quantum_codes/bivariate_bicycle_w5c",
    # "quantum_codes/bivariate_bicycle_w5_p3_a",
    "quantum_codes/bivariate_bicycle_paper",
    # "quantum_codes/surface_codes"
    # "quantum_rep_code/rep_code_overcomplete"
    # "quantum_rep_code/rep_code_overcomplete"
    ]
schedules = ["serial"]
decoders = ["lsd"]
local_search = [True]
local_search_lll = [True]
balance_weights = [True]

# TODO: I need to add another parameter for reduced / non-reduced.

for path in paths:
    # search_radius_plot(
    #     path,
    #     ds = [5, 7, 9, 11, 13]
    # )
    threshold_plot(
    # plot_paper(
        path,
        local_search=local_search,
        decoders=decoders,
        schedules=schedules,
        balance_weights=balance_weights,
        local_search_lll=local_search_lll,
        sphere_decoding=False,
        k="_",
        printdata=False,
    )
    # error_rate_scaling_plot(
    #     path,
    #     local_search=local_search,
    #     decoders=decoders,
    #     schedules=schedules,
    #     local_search_lll=local_search_lll,
    # )
# 