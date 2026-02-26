from typing import override

import numpy as np

from shapley_mcis_mes.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface, GameInterface


class SMES(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "S-MES"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        tau = int(np.ceil(T / (game.n + 1)))

        shapley_values = np.zeros(game.n)
        N = np.arange(game.n, dtype=int)

        q_quantiles = [(j / tau, (j + 1) / tau) for j in range(tau)]
        n_samples_used = 0

        for lower_bound, upper_bound in q_quantiles:
            q = np.random.uniform(lower_bound, upper_bound)
            S = N[np.random.binomial(1, q, size=game.n) == 1]
            base_eval = game.v(S)
            n_samples_used += 1
            for i in N:
                if i in S:
                    shapley_values[i] += base_eval - game.v(S[i != S])
                else:
                    shapley_values[i] += game.v(np.append(S, i)) - base_eval
                n_samples_used += 1

        shapley_values = shapley_values / tau

        check_number_of_samples_used(
            n_samples_used, T, SMES.name(), max_deviation=game.n + 1
        )

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
