import torch
import torch.nn as nn
from pathlib import Path
import json
import yaml

from typing import Literal
from os import PathLike

import networks

class ModelBuilder:

    def activation(activation_type: str) -> nn.Module:
        activations = {"ReLU": nn.ReLU(), 
                       "Tanh": nn.Tanh(),
                       "Sigmoid": nn.Sigmoid()}
        return activations[activation_type]

    def MLP(mlp_dict: dict) -> networks.MLP:

        activation = ModelBuilder.activation(mlp_dict["activation"])
        widths = mlp_dict["widths"]

        return networks.MLP(widths, activation)
    
    def Sequential(model_dicts: list[dict]) -> nn.Sequential:

        return nn.Sequential(*(build_model(model_dict)
                                for model_dict in model_dicts))
    
    def Normalizer(normalizer_dict: dict) -> networks.Normalizer:

        x_mean = torch.Tensor(normalizer_dict["x_mean"])
        x_var = torch.Tensor(normalizer_dict["x_var"])

        return networks.Normalizer(x_mean, x_var)
    
    def InverseNormalizer(normalizer_dict: dict) -> networks.InverseNormalizer:

        x_mean = torch.Tensor(normalizer_dict["x_mean"])
        x_var = torch.Tensor(normalizer_dict["x_var"])

        return networks.InverseNormalizer(x_mean, x_var)


class ModelLoader:

    def __new__(cls, model_dir: PathLike, load_state_dict: bool = True,
                mode: Literal["yaml", "json"] = "yaml") -> nn.Module:
        
        return load_model(model_dir, load_state_dict=load_state_dict, mode=mode)


def build_model(model_dict: dict) -> nn.Module:

    assert len(model_dict.keys()) == 1
    key = next(iter(model_dict.keys()))
    val = model_dict[key]

    model: nn.Module = getattr(ModelBuilder, key)(val)

    return model


def load_model(model_dir: PathLike, load_state_dict: bool = True,
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
    model = build_model(model_dict)

    if load_state_dict:
        model.load_state_dict(torch.load(model_dir / "state_dict.pt"))

    return model
