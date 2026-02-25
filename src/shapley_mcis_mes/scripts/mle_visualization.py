import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

n: int = 21
Qs: list[int] = [5, 1000]
tau_q: int = 2
yticks: dict[int, list[float]] = {5: [0, 0.5, 1, 1.5], 1000: [0, 25, 50, 75]}


def mle_visualization() -> None:
    for Q in Qs:
        s_values = np.arange(0, n)
        q_values = np.linspace(0, 1, Q)
        pmf_matrix = np.array([binom.pmf(s_values, n - 1, q) for q in q_values])
        bottom = np.zeros_like(s_values, dtype=float)

        plt.figure(figsize=(6.4, 4.8))
        for i in range(len(q_values)):
            plt.bar(s_values, pmf_matrix[i] * tau_q, bottom=bottom)

            if Q in yticks:
                plt.yticks(yticks[Q])

            bottom += pmf_matrix[i] * tau_q

        plt.xlabel(r"$s$")
        plt.ylabel(r"expected sample size")
        plt.grid(True)
        plt.savefig(f"figures/mle_visualization_{Q}.png", dpi=600)
        plt.show()
