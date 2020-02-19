import random

import numpy as np
import torch
from IPython.display import display, HTML
from rdkit import rdBase


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def clear_torch(model=None):
    if model:
        del model
    torch.cuda.empty_cache()


def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')


def header_str(a_str, n=80):
    """Returns a string formatted as a header."""
    return '{{:=^{:d}}}'.format(n).format(' ' + a_str + ' ')


def header_html(a_str, level=1):
    """Returns a string formatted as a header."""
    return display(HTML(f'<h{level}>{a_str}</h{level}>'))


def subset_list(alist, indices):
    return [alist[index] for index in indices]
