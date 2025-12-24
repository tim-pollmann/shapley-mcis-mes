import click
import numpy as np
import pandas as pd

from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface, GameInterface


def run_bias_comparison(
    game: GameInterface,
    ground_truth_shapley_values: np.ndarray,
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    taus: list[int],
    player: int,
    iters_per_tau: int = 5000,
) -> None:
    click.echo(f'Start experiment "{experiment_name}".')

    records: list[dict] = []

    for tau in taus:
        click.echo(f"{tau=}")
        for algorithm in algorithms:
            biases = []
            for _ in range(iters_per_tau):
                estimate = algorithm.run(game, tau)
                biases.append(estimate[player] - ground_truth_shapley_values[player])

            records.append(
                {
                    "tau": tau,
                    "algorithm": algorithm.name(),
                    "avg_bias": float(np.mean(biases)),
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"data/{experiment_name}.csv")
    click.echo(f'Finished experiment "{experiment_name}".')
