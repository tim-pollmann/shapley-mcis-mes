from itertools import combinations
from math import comb
from typing import override

import click
import numpy as np

from shapley_mcis_mes.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_mcis_mes.interfaces import ApproxAlgorithmInterface, GameInterface


class SMCIS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "S-MCIS"

    @staticmethod
    @override
    def run(game: GameInterface, tau: int) -> np.ndarray:
        N = np.arange(game.n, dtype=int)
        tau_s = int(np.ceil(tau / ((game.n + 1) ** 2)))

        n_samples_used = 0

        n = game.n
        stratum_estimators = np.zeros(shape=(n, n))
        stratum_counts = np.zeros(shape=(n, n))

        for s in range(0, game.n + 1):
            for _ in range(tau_s):
                S_bin = np.zeros(game.n, dtype=int)
                S_bin[:s] = 1
                np.random.shuffle(S_bin)
                S = N[S_bin == 1]
                base_eval = game.v(S)
                n_samples_used += 1

                for i in N:
                    if i in S:
                        stratum_estimators[i, s - 1] += base_eval - game.v(S[i != S])
                        stratum_counts[i, s - 1] += 1
                    else:
                        stratum_estimators[i, s] += game.v(np.append(S, i)) - base_eval
                        stratum_counts[i, s] += 1
                    n_samples_used += 1

        check_number_of_samples_used(
            n_samples_used,
            tau,
            SMCIS.name(),
            max_deviation=(game.n + 1) ** 2,
        )

        if (stratum_counts == 0).any():
            return np.full(n, np.nan)

        shapley_values = (stratum_estimators / stratum_counts).sum(axis=1) / n

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, tau: int, _: np.ndarray) -> np.ndarray:
        n = game.n
        N = list(range(n))
        squared_values = np.zeros(shape=(n, n))
        true_values = np.zeros(shape=(n, n))

        for s in range(0, n):
            for S in combinations(N, s):
                for i in N:
                    if i in S:
                        continue

                    S = np.array(S, dtype=int)
                    S_with_i = np.append(S, [i])
                    squared_values[i, s] += (
                        1 / comb(n - 1, s) * ((game.v(S_with_i) - game.v(S)) ** 2)
                    )
                    true_values[i, s] += (
                        1 / comb(n - 1, s) * (game.v(S_with_i) - game.v(S))
                    )

        tau_s = int(np.ceil(tau / ((game.n + 1) ** 2)))
        effective_tau_s = tau_s * (n + 1) / n
        strata_variances = 1 / effective_tau_s * (squared_values - true_values**2)
        variances = np.sum(strata_variances, axis=1) / (n**2)

        click.echo(
            click.style(
                "Warning: These are only approximated theoretical variances.",
                fg="yellow",
            )
        )

        return variances
