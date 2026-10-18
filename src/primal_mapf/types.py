"""Named, tuple-compatible interfaces shared by both networks."""
from typing import NamedTuple
from torch import Tensor


class NetworkOutput(NamedTuple):
    policy: Tensor
    value: Tensor
    hidden: Tensor
    cell: Tensor
    blocking: Tensor
    on_goal: Tensor
    legal_prediction: Tensor
