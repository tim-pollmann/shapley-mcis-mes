from shapley_mcis_mes.approx_algorithms.mcis import MCIS
from shapley_mcis_mes.approx_algorithms.mes import MES
from shapley_mcis_mes.approx_algorithms.os import OS
from shapley_mcis_mes.approx_algorithms.s_mcis import SMCIS
from shapley_mcis_mes.approx_algorithms.s_mes import SMES
from shapley_mcis_mes.exact_algorithms.brute_force_calculation_via_sum import (
    brute_force_calculation_via_sum,
)
from shapley_mcis_mes.games.airport_game import CustomAirportGame
from shapley_mcis_mes.games.explainability_game import (
    DiabetesGBRGame,
    HousingMLPGame,
    WineRFGame,
)
from shapley_mcis_mes.games.weighted_voting_game import CustomWeightedVotingGame
from shapley_mcis_mes.scripts.utils.run_comparison import run_comparison
from shapley_mcis_mes.scripts.utils.show_comparison import show_comparison
from shapley_mcis_mes.scripts.utils.update_plt_params import update_plt_params
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface

ALGORITHMS: list[ApproxAlgorithmInterface] = [OS, MES, SMES, MCIS, SMCIS]

TAUS_SYNTHETIC_GAMES = [30000, 40000, 50000, 60000, 70000, 80000, 100000]
TAUS_EXPLAINABILITY_GAMES = [10000, 15000, 20000, 25000, 30000, 40000, 50000]


def ag() -> None:
    update_plt_params()
    experiment_name = "ag"
    game = CustomAirportGame()
    ground_truth_shapley_values = game.shapley_values
    run_comparison(
        game,
        ground_truth_shapley_values,
        ALGORITHMS,
        experiment_name,
        TAUS_SYNTHETIC_GAMES,
    )
    show_comparison(experiment_name)


def wvg() -> None:
    update_plt_params()
    experiment_name = "wvg"
    game = CustomWeightedVotingGame()
    ground_truth_shapley_values = game.shapley_values
    run_comparison(
        game,
        ground_truth_shapley_values,
        ALGORITHMS,
        experiment_name,
        TAUS_SYNTHETIC_GAMES,
    )
    show_comparison(experiment_name)


def diabetes() -> None:
    update_plt_params()
    experiment_name = "diabetes"
    game = DiabetesGBRGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)
    run_comparison(
        game,
        ground_truth_shapley_values,
        ALGORITHMS,
        experiment_name,
        TAUS_EXPLAINABILITY_GAMES,
    )
    show_comparison(experiment_name)


def housing() -> None:
    update_plt_params()
    experiment_name = "housing"
    game = HousingMLPGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)
    run_comparison(
        game,
        ground_truth_shapley_values,
        ALGORITHMS,
        experiment_name,
        TAUS_EXPLAINABILITY_GAMES,
    )
    show_comparison(experiment_name)


def wine() -> None:
    update_plt_params()
    experiment_name = "wine"
    game = WineRFGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)
    run_comparison(
        game,
        ground_truth_shapley_values,
        ALGORITHMS,
        experiment_name,
        TAUS_EXPLAINABILITY_GAMES,
    )
    show_comparison(experiment_name)
