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
    config['train_lr'] = 0.0005705023
    config['train_clip_grad'] = 20
    return config


def vae_config(rand=False):
    config = AttributeDict()

    config['enc_hidden_dim'] = 768
    config['enc_n_layers'] = 1
    config['enc_dropout'] = 0.24

    config['dec_hidden_dim'] = 704
    config['dec_n_layers'] = 1
    config['dec_dropout'] = 0.24

    config["kl_cycle_length"] = 15
    config["kl_cycle_constant"] = 3
    config["kl_weight_start"] = 1e-5
    config["kl_weight_end"] = 0.002794217

    config['vae_latent_dim'] = 288
    config['vae_y_dec'] = True
    config['vae_selfies_dec'] = True
    config['vae_mof_enc'] = True
    config['vae_mof_dec'] = True
    config['vae_duplicate_smiles'] = True

    return config


def mof_config(rand=False):
    config = AttributeDict()
    config['mof_encoding'] = 'cat'
    config['mof_weighted_loss'] = True
    config['mof_w_start'] = 0.0
    config['mof_w_end'] = 0.001
    config['mof_start'] = 0
    config['mof_const_length'] = 10
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
    config['y_w_end'] = 0.1
    config['y_start'] = 0
    config['y_const_length'] = 10
    config['scaler_type'] = 'standard'

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
        config['mof_encoding'] = 'cats'
        config['y_w_start'] = 1.0
        config['y_w_end'] = 1.0
        config['kl_weight_start'] = 0.0
        config['kl_weight_end'] = 0.0
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
