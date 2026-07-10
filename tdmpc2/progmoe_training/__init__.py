"""Progressive-MoE continual learning for PRISM-WM TD-MPC2.

Architecture:
  - `MaskedMoEBlock`: static-K MoE with `_active_K` mask. No forward
    or gradient through masked experts.
  - `PrismBackbone`: shared encoder + dynamics MoE. Persists across tasks.
  - `TaskModules`: per-task reward / Q / termination / π / scale. Fresh
    at every task boundary; has NO reference back to the backbone.
  - `EvalAgent`: pairs a shared backbone with a loaded TaskModules for
    cross-task evaluation (DreamerV3 pattern).
  - `ContinualTDMPC2`: training agent. Owns backbone + current task_modules
    + three optimizers; saves/loads them as two separate checkpoint files.
"""

from progmoe_training.masked_moe import MaskedMoEBlock
from progmoe_training.backbone import PrismBackbone, TaskModules, EvalAgent
from progmoe_training.continual_tdmpc2 import ContinualTDMPC2

__all__ = [
    'MaskedMoEBlock',
    'PrismBackbone',
    'TaskModules',
    'EvalAgent',
    'ContinualTDMPC2',
]
