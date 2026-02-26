from shapley_mcis_mes.approx_algorithms.mes import MES
from shapley_mcis_mes.approx_algorithms.os import OS
from shapley_mcis_mes.approx_algorithms.s_mes import SMES
from shapley_mcis_mes.games.airport_game import CustomAirportGameSmall
from shapley_mcis_mes.scripts.utils.plot_bias_comparison import plot_bias_comparison
from shapley_mcis_mes.scripts.utils.run_bias_comparison import run_bias_comparison
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [OS, MES, SMES]
_Ts = [30000, 40000, 50000, 60000, 70000, 80000, 100000]
_PLAYER: int = 0
_ITERS_PER_T = 5000


def default() -> None:
    experiment_name = "bias_comparison"
    game = CustomAirportGameSmall()
    ground_truth_shapley_values = game.shapley_values

    run_bias_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _Ts,
        _PLAYER,
        _ITERS_PER_T,
    )

    plot_bias_comparison(experiment_name, _PLAYER)
