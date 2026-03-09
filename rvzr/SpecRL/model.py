import torch
import torch.nn as nn
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override


def _get_flat_input_size(obs_space, seq_size=100, num_inputs=20):
    if isinstance(obs_space, spaces.Dict):
        from ray.rllib.utils.spaces.space_utils import flatten_space
        flat_space = flatten_space(obs_space)
        return flat_space.shape[0]
    return obs_space.shape[0]


class CustomMLPBackbone(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        custom_config = model_config.get("custom_model_config", {})
        hidden_sizes = custom_config.get("hidden_sizes", [512, 256, 128])
        activation = custom_config.get("activation", "relu")
        use_layer_norm = custom_config.get("use_layer_norm", False)
        dropout = custom_config.get("dropout", 0.0)

        input_size = _get_flat_input_size(obs_space)

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_activation_module(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h

        self.backbone = nn.Sequential(*layers)
        self._features = None

        self.policy_head = nn.Linear(prev_size, num_outputs)
        self.value_head = nn.Linear(prev_size, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict.get("obs_flat", input_dict["obs"])
        if isinstance(obs, dict):
            obs = torch.cat([v.float().flatten(1) for v in obs.values()], dim=1)
        else:
            obs = obs.float()

        self._features = self.backbone(obs)
        return self.policy_head(self._features), state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None
        return self.value_head(self._features).squeeze(-1)


def _activation_module(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.ReLU()


def register_custom_models():
    ModelCatalog.register_custom_model("CustomMLPBackbone", CustomMLPBackbone)
