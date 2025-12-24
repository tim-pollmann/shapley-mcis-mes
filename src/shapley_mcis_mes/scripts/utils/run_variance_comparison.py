import click
import numpy as np
import pandas as pd

from shapley_mcis_mes.exact_algorithms.brute_force_calculation_via_sum import (
    brute_force_calculation_via_sum,
)
from shapley_mcis_mes.games.weighted_voting_game import WeightedVotingGame
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface

_WEIGHTS = [1, 2, 3, 1, 1, 1, 1]
_GAMES = {
    quota + 1: WeightedVotingGame(_WEIGHTS, quota + 1) for quota in range(sum(_WEIGHTS))
}


def run_variance_comparison(
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    tau: int,
    player: int,
    n_iters: int = 5,
) -> None:
    click.echo(f'Start experiment "{experiment_name}".')

    records: list[dict] = []

    for quota, game in _GAMES.items():
        click.echo(f"Current quota under study: {quota}")
        true_phi = brute_force_calculation_via_sum(game)

        for algorithm in algorithms:
            name = algorithm.name()

            theor_var = algorithm.variance(game, tau, true_phi)
            records.append(
                {
                    "quota": quota,
                    "algorithm": name,
                    "variance_type": "theoretical",
                    "variance": float(theor_var[player]),
                }
            )

            runs = []
            for _ in range(n_iters):
                est = algorithm.run(game, tau)
                runs.append((est[player] - true_phi[player]) ** 2)

            records.append(
                {
                    "quota": quota,
                    "algorithm": name,
                    "variance_type": "empirical",
                    "variance": float(np.mean(np.array(runs))),
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"data/{experiment_name}.csv", index=False)
    click.echo(f'Finished experiment "{experiment_name}".')
