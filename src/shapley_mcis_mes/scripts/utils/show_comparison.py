import matplotlib.pyplot as plt
import pandas as pd


def show_comparison(
    experiment_name: str, yticks: list[float] = None, ylim: float = None
) -> None:
    df = pd.read_csv(f"data/{experiment_name}.csv", index_col="tau")

    plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))

    if yticks is not None:
        plt.yticks(yticks)

    if ylim:
        plt.ylim(top=ylim)

    plt.xlabel(r"overall sample budget $T$")
    plt.ylabel(r"mean squared error of $\hat{\phi}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"figures/{experiment_name}.png", dpi=600)

    plt.show()
