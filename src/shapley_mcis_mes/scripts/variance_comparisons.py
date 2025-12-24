from shapley_mcis_mes.approx_algorithms.mcis import MCIS
from shapley_mcis_mes.approx_algorithms.mes import MES
from shapley_mcis_mes.approx_algorithms.s_mcis import SMCIS
from shapley_mcis_mes.scripts.utils.plot_variance_comparison import (
    plot_variance_comparison,
)
from shapley_mcis_mes.utils.interfaces import ApproxAlgorithmInterface

_TAU = 20000
_ALGORITHMS: list[ApproxAlgorithmInterface] = [MCIS, MES, SMCIS]
_PLAYER = 0


def default() -> None:
    experiment_name = "variance_comparison"
    # run_variance_comparison(_ALGORITHMS, experiment_name, _TAU, _PLAYER, n_iters=1)

    plot_variance_comparison(experiment_name, _PLAYER)
