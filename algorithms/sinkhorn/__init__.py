from sinkhorn import Sinkhorn
from .prob_sinkhorn import ProbSinkhorn
from .penalty_sinkhorn import PenaltySinkhorn
from .lassosinkhorn import LassoSinkhorn
from .l1_sinkhorn import L1Sinkhorn
from .recover_SWaT import recover_SWaT
from .kl_sinkhorn import KLSinkhorn

__all__ = [
    "Sinkhorn",
]

__version__ = "0.0.1"