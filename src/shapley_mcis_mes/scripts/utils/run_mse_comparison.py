import click
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface, GameInterface


def run_mse_comparison(
    game: GameInterface,
    ground_truth_shapley_values: np.ndarray,
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    taus: list[int],
    iters_per_tau: int = 250,
) -> None:
    click.echo(f'Start experiment "{experiment_name}".')

    mses = {algorithm.name(): [] for algorithm in algorithms}

    for tau in taus:
        click.echo(f"{tau=}")
        mse_tmp = {algorithm.name(): [] for algorithm in algorithms}

        for _ in range(iters_per_tau):
            for algorithm in algorithms:
                shapley_values = algorithm.run(game, tau)

                if not np.isnan(shapley_values).any():
                    mse = mean_squared_error(
                        ground_truth_shapley_values, shapley_values
                    )
                    mse_tmp[algorithm.name()].append(mse)

        for algorithm in algorithms:
            # make sure that algorithms like S-MCIS have at least half the amount of executions in comparison to all
            # other algorithms
            if len(mse_tmp[algorithm.name()]) > iters_per_tau // 2:
                avg_mse = float(np.mean(mse_tmp[algorithm.name()]))
                mses[algorithm.name()].append(avg_mse)
            else:
                mses[algorithm.name()].append(np.nan)

    for algorithm in algorithms:
        mses[algorithm.name()] = mses.pop(algorithm.name())

    df = pd.DataFrame(mses, index=taus)
    df.index.name = "tau"
    df.to_csv(f"data/{experiment_name}.csv")
    click.echo(f'Finished experiment "{experiment_name}".')
