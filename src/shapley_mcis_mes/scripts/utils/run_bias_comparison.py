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
    iters_per_tau: int,
) -> None:
    click.echo(f'Start experiment "{experiment_name}".')

    records: list[dict] = []

    for tau in taus:
        click.echo(f"{tau=}")

        for algorithm in algorithms:
            biases = []

            for _ in range(iters_per_tau):
                approximated_shapley_values = algorithm.run(game, tau)

                if not np.isnan(approximated_shapley_values).any():
                    biases.append(
                        approximated_shapley_values[player]
                        - ground_truth_shapley_values[player]
                    )

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
