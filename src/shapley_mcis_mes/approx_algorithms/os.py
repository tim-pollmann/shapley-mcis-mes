from typing import override

import numpy as np

from shapley_mcis_mes.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface, GameInterface


class OS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "OS"

    @staticmethod
    @override
    def run(game: GameInterface, tau: int, tau_per_q: int = 2) -> np.ndarray:
        shapley_values = np.zeros(game.n)
        N = np.arange(game.n, dtype=int)
        Q = int(np.ceil(tau / (game.n + 1) / tau_per_q))

        if Q < 2:
            raise ValueError(
                "Q should be >= 2. Please adapt m and m_per_q accordingly."
            )

        q_quantiles = [j / (Q - 1) for j in range(Q)]
        n_samples_used = 0

        for q in q_quantiles:
            for _ in range(tau_per_q):
                S = N[np.random.binomial(1, q, size=game.n) == 1]
                base_eval = game.v(S)
                n_samples_used += 1
                for i in N:
                    if i in S:
                        shapley_values[i] += base_eval - game.v(S[i != S])
                    else:
                        shapley_values[i] += game.v(np.append(S, i)) - base_eval
                    n_samples_used += 1

        shapley_values = shapley_values / (Q * tau_per_q)

        check_number_of_samples_used(
            n_samples_used,
            tau,
            OS.name(),
            max_deviation=tau_per_q * (game.n + 1),
        )

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, tau: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
