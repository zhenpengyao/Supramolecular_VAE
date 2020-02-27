import nujson as json
import os
from collections import Counter, OrderedDict
import numpy as np
from vaemof.utils import header_str

PRESETS = ['vae', 'ssvae', 'mof_vae', 'mof_y', 'full']

rand_uniform10 = lambda low, high: 10 ** np.random.uniform(low, high)
rand_uniform = lambda low, high: np.random.uniform(low, high)
rand_int = lambda low, high: int(np.random.randint(low, high + 1))
rand_choice = lambda x: np.random.choice(x)


def basic_config(work_dir, rand=False):
    config = AttributeDict()
    # Model saving and loading. If None, should be filed.
    config['files_data'] = 'data/MOF_gen_train.csv.gz'
    config['files_prop'] = 'data/MOF_properties_train.csv.gz'
    config['files_results'] = work_dir
    config['files_config'] = 'config.json'
    config['files_log'] = 'log.csv'
    # Training parameters.
    config['train_device'] = 'cpu'
    config['train_seed'] = 42
    config['train_batch_size'] = 1536
    config['train_epochs'] = 60
    config['train_lr'] = 0.00013138571209279765 #float(rand_uniform10(-5, -2)) if rand else 5e-3
    config['train_clip_grad'] = 20
    return config


def vae_config(rand=False):
    config = AttributeDict()

    config['enc_hidden_dim'] = 768 #int(rand_choice(np.arange(128, 808, 64))) if rand else 256
    config['enc_n_layers'] = 1 #int(rand_int(1, 3)) if rand else 1
    config['enc_dropout'] = 0.18 #float(rand_choice(np.linspace(0.0,.3,6))) if rand else 0.25

    config['dec_hidden_dim'] = 192 #int(rand_choice(np.arange(128, 808, 64))) if rand else 512
    config['dec_n_layers'] = 2 #int(rand_int(1, 3)) if rand else 1
    config['dec_dropout'] = 0.12 #float(rand_choice(np.linspace(0.0,.3,6))) if rand else 0.0

    config["kl_cycle_length"] = 15
    config["kl_cycle_constant"] = 3
    config["kl_weight_start"] = 1e-5
    config["kl_weight_end"] = 0.0012886751810776008 #float(rand_uniform10(-5, 2)) if rand else 0.77

    config['vae_latent_dim'] = 256 #int(rand_choice(np.arange(128, 316, 16))) if rand else 256
    config['vae_y_dec'] = True
    config['vae_selfies_dec'] = True
    config['vae_mof_enc'] = True
    config['vae_mof_dec'] = True
    config['vae_duplicate_smiles'] = True

    return config


def mof_config(rand=False):
    config = AttributeDict()
    config['mof_encoding'] = 'all' #rand_choice(['all', 'id', 'cats']) if rand else 'all'
    config['mof_weighted_loss'] = False #rand_choice([True, False]) if rand else True
    config['mof_w_start'] = 0.0
    config['mof_w_end'] = 0.5 #float(rand_choice([1e-3, 1e-2, .1, .25, .5, .75, 1.0])) if rand else 1.0
    config['mof_start'] = 15  # Epoch to start weight annealing.
    config['mof_const_length'] = 10  # How many epochs to have constant values.
    return config


def y_config(rand=False):
    config = AttributeDict()

    config['y_labels'] = ['lcd', 'pld',
                          'density', 'avf', 'avsa', 'agsa', 'co2n2_co2_mol_kg', 'co2n2_n2_mol_kg',
                          'co2n2_selectivity', 'co2n2_heat_avg', 'co2n2_heat_co2',
                          'co2n2_heat_n2', 'co2n2_heat_molfrac', 'co2ch4_co2_mol_kg',
                          'co2ch4_ch4_mol_kg', 'co2ch4_selectivity', 'co2ch4_heat_avg',
                          'co2ch4_heat_co2', 'co2ch4_heat_ch4', 'co2ch4_heat_molfrac', 'scscore']
    config['y_weights'] = [1] * len(config['y_labels'])
    config['y_w_start'] = 0.0
    config['y_w_end'] = 0.5 #float(rand_choice([1e-3, 1e-2, .1, .25, .5, .75, 1.0])) if rand else 1.0
    config['y_start'] = 15  # Epoch to start weight annealing.
    config['y_const_length'] = 10  # How many epochs to have constant values.

    return config


def get_model_config(work_dir, preset='full', rand=False):
    config = basic_config(work_dir, rand)
    config.update(vae_config(rand))
    config.update(mof_config(rand))
    config.update(y_config(rand))
    if preset == 'vae':
        config['vae_y_dec'] = False
        config['vae_selfies_dec'] = True
        config['vae_mof_enc'] = False
        config['vae_mof_dec'] = False
    elif preset == 'ssvae':
        config['vae_y_dec'] = True
        config['vae_selfies_dec'] = True
        config['vae_mof_enc'] = False
        config['vae_mof_dec'] = False
    elif preset == 'mof_y':
        config['vae_y_dec'] = True
        config['vae_selfies_dec'] = False
        config['vae_mof_enc'] = True
        config['vae_mof_dec'] = False
    elif preset == 'mof_vae':
        config['vae_y_dec'] = False
        config['vae_selfies_dec'] = True
        config['vae_mof_enc'] = True
        config['vae_mof_dec'] = True
    elif preset == 'full':
        pass
    else:
        raise ValueError(f'preset={preset} should be one of {PRESETS}.')
    return config


def at_results_dir(config, key):
    return os.path.join(config['files_results'], config[key])


def check_files(config):
    # Need to create directories if they don't exists.
    if not os.path.exists(config['files_results']):
        os.makedirs(config['files_results'])

    for key in ['data', 'prop']:
        if not os.path.exists(config[f'files_{key}']):
            print(f"Warning: File {config[f'files_{key}']} does not exist!")


def testing_config(config):
    if config['train_device'] == 'cuda':
        testing = False
    else:
        config['train_batch_size'] = 64
        config['train_epochs'] = 20
        testing = True
    return testing


class AttributeDict(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_namespace(cls, namespace):
        return cls(vars(namespace))

    def save_to_json(self, filename):
        with open(filename, 'w') as afile:
            json.dump(self, afile, indent=4)

    @classmethod
    def from_jsonfile(cls, filename):
        with open(filename, 'r') as afile:
            data = json.loads(afile.read())
        return cls(data)


def sort_config(config):
    prefixes = [key.split('_')[0] for key in config.keys()]
    count_prefixes = Counter(prefixes)
    count_prefixes = OrderedDict(count_prefixes.most_common())
    new_order = []
    for prefix in count_prefixes.keys():
        new_order = new_order + \
                    [key for key in config.keys() if key.split('_')[0] == prefix]
    sorted_config = AttributeDict([(key, config[key]) for key in new_order])
    return sorted_config


def print_config(config, title='config'):
    print(header_str(title))
    config = sort_config(config)
    last_prefix = ''
    for key, value in config.items():
        cur_prefix = key.split('_')[0]
        if cur_prefix != last_prefix:
            print(f'== {cur_prefix} == :')
            last_prefix = cur_prefix
        if value is not None:
            print('{:>20s}:{:>20s}'.format(key, str(value)))
