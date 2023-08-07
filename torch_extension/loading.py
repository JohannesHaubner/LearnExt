import torch
import torch.nn as nn
from pathlib import Path
import json
import yaml

from typing import Literal
from os import PathLike

from torch_extension.networks import MLP, Normalizer

class ModelBuilder:

    def __new__(cls, model_dict: dict) -> nn.Module:

        assert len(model_dict.keys()) == 1
        key = next(iter(model_dict.keys()))
        val = model_dict[key]

        model: nn.Module = getattr(cls, key)(val)

        return model
    
    def activation(activation_type: str) -> nn.Module:
        activations = {"ReLU": nn.ReLU(), 
                       "Tanh": nn.Tanh(),
                       "Sigmoid": nn.Sigmoid()}
        return activations[activation_type]

    def MLP(mlp_dict: dict) -> MLP:

        activation = ModelBuilder.activation(mlp_dict["activation"])
        widths = mlp_dict["widths"]

        return MLP(widths, activation)
    
    def Sequential(model_dicts: list[dict]) -> nn.Sequential:

        return nn.Sequential(*(ModelBuilder(model_dict)
                                for model_dict in model_dicts))
    
    def Normalizer(normalizer_dict: dict) -> Normalizer:

        x_mean = torch.Tensor(normalizer_dict["x_mean"])
        x_std = torch.Tensor(normalizer_dict["x_std"])

        return Normalizer(x_mean, x_std)


class ModelLoader:

    def __new__(cls, model_dir: PathLike, load_state_dict: bool = True,
                mode: Literal["yaml", "json"] = "yaml") -> nn.Module:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise ValueError("Non-existent directory.")

        if mode == "yaml":
            with open(model_dir / "model.yml", "r") as infile:
                model_dict = yaml.safe_load(infile.read())
        elif mode == "json":
            with open(model_dir / "model.json", "r") as infile:
                model_dict= json.loads(infile.read())
        else:
            raise ValueError
        model: nn.Module = ModelBuilder(model_dict)

        if load_state_dict:
            model.load_state_dict(torch.load(model_dir / "state_dict.pt"))

        return model
