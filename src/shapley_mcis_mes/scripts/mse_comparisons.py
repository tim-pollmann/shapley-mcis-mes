from shapley_mcis_mes.approx_algorithms.mcis import MCIS
from shapley_mcis_mes.approx_algorithms.mes import MES
from shapley_mcis_mes.approx_algorithms.os import OS
from shapley_mcis_mes.approx_algorithms.s_mcis import SMCIS
from shapley_mcis_mes.approx_algorithms.s_mes import SMES
from shapley_mcis_mes.exact_algorithms.brute_force_calculation_via_sum import (
    brute_force_calculation_via_sum,
)
from shapley_mcis_mes.games.airport_game import CustomAirportGameLarge
from shapley_mcis_mes.games.explainability_game import (
    DiabetesGBRGame,
    HousingMLPGame,
    WineRFGame,
)
from shapley_mcis_mes.games.weighted_voting_game import CustomWeightedVotingGame
from shapley_mcis_mes.scripts.utils.plot_mse_comparison import plot_mse_comparison
from shapley_mcis_mes.scripts.utils.run_mse_comparison import run_mse_comparison
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [OS, MES, SMES, MCIS, SMCIS]

_TAUS_SYNTHETIC_GAMES = [30000, 40000, 50000, 60000, 70000, 80000, 100000]
_TAUS_EXPLAINABILITY_GAMES = [10000, 15000, 20000, 25000, 30000, 40000, 50000]


def ag() -> None:
    experiment_name = "ag"
    game = CustomAirportGameLarge()
    ground_truth_shapley_values = game.shapley_values

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_SYNTHETIC_GAMES,
    )
    plot_mse_comparison(experiment_name)


def wvg() -> None:
    experiment_name = "wvg"
    game = CustomWeightedVotingGame()
    ground_truth_shapley_values = game.shapley_values

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_SYNTHETIC_GAMES,
    )
    plot_mse_comparison(experiment_name)


def diabetes() -> None:
    experiment_name = "diabetes"
    game = DiabetesGBRGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
    )
    plot_mse_comparison(experiment_name)


def housing() -> None:
    experiment_name = "housing"
    game = HousingMLPGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
    )
    plot_mse_comparison(experiment_name)


def wine() -> None:
    experiment_name = "wine"
    game = WineRFGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
    )
    plot_mse_comparison(experiment_name)
