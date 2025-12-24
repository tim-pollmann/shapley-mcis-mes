import pandas as pd
from matplotlib import pyplot as plt

from shapley_mcis_mes.scripts.utils.update_plt_params import update_plt_params


def plot_bias_comparison(experiment_name: str, player: int) -> None:
    update_plt_params()

    df = pd.read_csv(f"data/{experiment_name}.csv")

    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values("tau")
        plt.plot(
            group["tau"],
            group["avg_bias"],
            label=algorithm,
        )

    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    plt.xlabel(r"overall sample budget $T$")
    plt.ylabel(r"bias of $\hat{\phi}_" + f"{player}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{experiment_name}.png", dpi=600)
    plt.show()
