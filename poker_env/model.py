"""Custom RLlib model with action masking for poker.

An Actor-Critic MLP registered as a custom model.  The Dict observation
contains an ``"action_mask"`` key which is applied to the logits so the
policy never samples illegal actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

FLOAT_MIN = -1e38


class ActionMaskModel(TorchModelV2, nn.Module):
    """Actor-Critic MLP that reads action_mask from the Dict observation."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {})
        hidden = custom.get("hidden", 256)

        orig_space = getattr(obs_space, "original_space", obs_space)
        obs_dim = orig_space["observation"].shape[0]

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_outputs),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self._value: torch.Tensor | None = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"].float()
        mask = input_dict["obs"]["action_mask"].float()

        h = self.trunk(obs)
        logits = self.actor(h)
        self._value = self.critic(h).squeeze(-1)

        inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
        logits = logits + inf_mask

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "forward() must be called first"
        return self._value
