import numpy as np
from typing import Dict, List, Tuple
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )


        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        for out_size in post_fcnet_hiddens:
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation      ,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size

        self._convs = nn.Sequential(*layers)

        # Last layer is logits layer.
        self._logits = SlimFC(
            in_size=in_size,
            out_size=num_outputs,
            activation_fn=None,
            initializer=normc_initializer(0.01),
        )

        # Build the value layers
        self._value_branch = SlimFC(
                in_size, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        # Holds the current "base" output (before logits layer).
        self._features = None


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        obs = obs.permute(0, 3, 1, 2)
        conv_out = self._convs(obs)
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out

        logits = self._logits(conv_out)

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        features = self._features
        value = self._value_branch(features)
        return value.squeeze(1)  # [B,1] -> [B]


class VisionNetwork2(VisionNetwork):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        actor_fc_hiddens=[],
        critic_fc_hiddens=[],
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(
            filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        # actor_fc_hiddens = model_config.get("actor_fc_hiddens", [])
        # critic_fc_hiddens = model_config.get("critic_fc_hiddens", [])

        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )

        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        for out_size in post_fcnet_hiddens:
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size

        self._convs = nn.Sequential(*layers)

        # Last layer is logits layer.

        actor_fc_in_size = in_size
        actor_layers = []
        for out_size in actor_fc_hiddens:
            actor_layers.append(
                SlimFC(
                    in_size=actor_fc_in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation,
                    initializer=normc_initializer(0.01),
                )
            )
            actor_fc_in_size = out_size

        actor_layers.append(SlimFC(
            in_size=actor_fc_in_size,
            out_size=num_outputs,
            activation_fn=None,
            initializer=normc_initializer(0.01),
        ))

        self._logits = nn.Sequential(*actor_layers)

        # Build the value layers
        critic_fc_in_size = in_size
        critic_layers = []
        for out_size in critic_fc_hiddens:
            critic_layers.append(
                SlimFC(
                    in_size=critic_fc_in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation,
                    initializer=normc_initializer(0.01),
                )
            )
            critic_fc_in_size = out_size
        critic_layers.append(SlimFC(
            in_size=critic_fc_in_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None
        ))

        self._value_branch = nn.Sequential(*critic_layers)
        # Holds the current "base" output (before logits layer).
        self._features = None