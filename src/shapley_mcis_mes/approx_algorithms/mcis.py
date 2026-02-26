from math import factorial
from typing import override

import numpy as np

from shapley_mcis_mes.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_mcis_mes.exact_algorithms.utils.powerset_iterator import (
    powerset_iterator,
)
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface, GameInterface


class MCIS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "MCIS"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        tau = int(np.ceil(T / (game.n + 1)))

        shapley_values = np.zeros(game.n)
        N = np.arange(game.n, dtype=int)

        n_samples_used = 0

        for _ in range(tau):
            s = np.random.randint(0, game.n + 1)
            S = np.random.choice(N, size=s, replace=False)
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
            n_samples_used,
            T,
            MCIS.name(),
            max_deviation=game.n + 1,
        )

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        tau = int(np.ceil(T / (game.n + 1)))

        n = game.n
        variances = np.zeros(n)

        for S in powerset_iterator(range(n)):
            s = len(S)

            v_S = game.v(S)

            for i in range(n):
                if i not in S:
                    S_with_i = np.append(S, [i])
                    variances[i] += (
                        factorial(s)
                        * factorial(n - s - 1)
                        / factorial(n)
                        * ((game.v(S_with_i) - v_S) ** 2)
                    )

        return (variances - (true_values**2)) / tau
