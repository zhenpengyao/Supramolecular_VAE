from collections import OrderedDict
from typing import Text, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display

from vaemof.configs import at_results_dir
from vaemof.utils import header_html


def make_mlp(layer_dims: List[int], act: Text = 'relu', batchnorm: bool = False, dropout: float = 0.0,
             activation_last: bool = False) -> nn.Sequential:
    layers = []
    n_layers = len(layer_dims)
    assert n_layers >= 2, "MLP should be at least 2 layers."
    last_layer = n_layers - 2
    for index in range(last_layer + 1):
        layers.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
        if index < last_layer:
            layers.append(get_activation(act))
        elif activation_last:
            layers.append(get_activation(act))
        if batchnorm:
            layers.append(nn.BatchNorm1d(layer_dims[index + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def module_to_row(name, module, head=None):
    """Convert a module into a pd.Dataframe row."""
    module_dict = OrderedDict()
    head = f'{head}.' if head else ''
    module_dict['Name'] = f'{head}{name}'
    module_dict['Module'] = module.__class__.__name__
    module_dict['Extra'] = module.extra_repr()
    module_dict['submodule'] = True if head else False
    module_dict['trainable'] = any(p.requires_grad for p in module.parameters())
    module_dict['n_params'] = sum([np.product(p.size()) for p in module.parameters()])
    module_dict['trainable_params'] = sum([np.product(p.size()) for p in module.parameters() if p.requires_grad])
    return module_dict


def model_summary(model, include_children=True):
    """"Print a pytorch model."""
    header_html(model.__class__.__name__)
    modules = model.named_children()
    rows = []
    for name, module in modules:
        rows.append(module_to_row(name, module))
        if include_children:
            for sub_name, sub_module in module.named_children():
                rows.append(module_to_row(sub_name, sub_module, name))

    model_df = pd.DataFrame(rows)
    display(model_df)
    n_params = sum([np.product(p.size()) for p in model.parameters()])
    train_params = sum([np.product(p.size()) for p in model.parameters() if p.requires_grad])
    print(f'Trainable params: {train_params} out of {n_params} total ({train_params / n_params * 100.0}%)')


def get_activation(astr: Text) -> nn.modules.activation:
    activations = {'selu': nn.SELU, 'relu': nn.ReLU, 'prelu': nn.PReLU, 'leaky_relu': nn.LeakyReLU,
                   'softplus': nn.Softplus}
    return activations[astr]()


def save_model(model):
    model = model.to('cpu')
    torch.save(model.state_dict(), at_results_dir(model.hparams, 'files_model'))
    torch.save(model.vocab, at_results_dir(model.hparams, 'files_vocab'))
    model.hparams.save_to_json(at_results_dir(model.hparams, 'files_config'))
