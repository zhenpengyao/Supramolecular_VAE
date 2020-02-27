import multiprocessing
from collections import OrderedDict
from typing import Any, Text, List, Optional, Tuple

import numpy as np
import pandas as pd
import selfies
import torch
from joblib import Parallel, delayed
from more_itertools import chunked
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, QuantileTransformer
from tqdm.auto import tqdm


def isosmiles(x):
    mol = Chem.MolFromSmiles(x)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    else:
        return x


def valid_smiles(smiles):
    if len(smiles) == 0:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


class SS:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


EXTRA_PAD = 5


class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)
        # extra padding.
        max_len = np.max([len(string) for string in data]) + EXTRA_PAD
        return cls(chars, max_len=max_len, *args, **kwargs)

    def __init__(self, chars, max_len=None, ss=SS):
        self.ss = ss
        self.pad_char = ss.pad
        self.bos_char = ss.bos
        self.eos_char = ss.eos
        self.unk_char = ss.unk
        self.max_len = max_len
        all_syms = sorted(list(chars)) + \
                   [self.pad_char, self.bos_char, self.eos_char]
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def pad(self):
        return self.c2i[self.pad_char]

    @property
    def bos(self):
        return self.c2i[self.bos_char]

    @property
    def eos(self):
        return self.c2i[self.eos_char]

    def pad_ids(self, ids, seq_len):
        n = len(ids)
        if n > seq_len:
            raise ValueError(f'{seq_len} < {n}!')
        return ids + [self.pad] * (seq_len - n)

    def unpad_ids(self, ids):
        return [id for id in ids if id != self.pad]

    def correct_sequence(self, ids):
        n_bos = sum([i == self.bos for i in ids])
        n_eos = sum([i == self.bos for i in ids])
        if n_bos != 1 or n_eos != 1:
            return False
        bos_at_start = ids[0] == self.bos
        eos_index = np.argmax([i == self.eos for i in ids])
        just_padding = all([i == self.pad for i in ids[eos_index + 1:]])
        return bos_at_start and just_padding

    def remove_bos_eos(self, ids):
        return ids[1:-1]

    def add_bos_eos(self, ids):
        return [self.bos] + ids + [self.eos]

    def char_to_id(self, char):
        if char not in self.c2i:
            raise ValueError(f'{char} not in dictionary!')

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            raise ValueError(f'{id} not in vocab.')
        return self.i2c[id]

    def string_preprocess(self, astr):
        return astr

    def string_to_ids(self, string, pad_seq=False, max_len=0):
        ids = [self.char_to_id(c) for c in self.string_preprocess(string)]
        max_len = self.max_len if max_len == 0 else max_len
        ids = self.add_bos_eos(ids)
        ids = self.pad_ids(ids, max_len) if pad_seq else ids
        return ids

    def charlist_postprocess(self, char_list):
        return ''.join(char_list)

    def ids_to_string(self, ids, remove_padding=True):
        if len(ids) == 0:
            return ''
        correct = self.correct_sequence(ids)
        if correct:
            ids = self.unpad_ids(ids) if remove_padding else ids
            ids = self.remove_bos_eos(ids)
            char_list = [self.id2char(id) for id in ids]
            return self.charlist_postprocess(char_list)
        else:
            return ''.join([self.id2char(id) for id in ids])

    def df_to_ids(self, df, column, batch=100000):
        n = len(df)
        n_loops = int(np.ceil(n / batch))
        n_jobs = multiprocessing.cpu_count()
        disable = n_loops < 5
        ids_list = []
        for indexes in tqdm(chunked(range(n), batch), total=n_loops, disable=disable, desc='SMILES'):
            strings = df.iloc[list(indexes)][column].tolist()
            ids = Parallel(n_jobs=n_jobs)(delayed(self.string_to_ids)(s) for s in strings)
            ids_list.extend(ids)
        return ids_list


class OneHotVocab(CharVocab):
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))


class SELFIESVocab(OneHotVocab):

    @classmethod
    def from_data(cls, smiles_list, *args, **kwargs):
        alphabet = set()

        def update_alphabet(x):
            selfies_list = cls.smiles_to_selfies_list(x)
            return (set(selfies_list), len(selfies_list))

        n_jobs = multiprocessing.cpu_count()
        uniq_chars, seq_lens = zip(*Parallel(n_jobs=n_jobs)(
            delayed(update_alphabet)(string) for string in tqdm(smiles_list)))
        max_len = np.max(seq_lens)
        for aset in uniq_chars:
            alphabet.update(aset)
        print(f'Alphabet size is {len(alphabet) + 4}')
        print(f'Max seq length is {max_len} with {EXTRA_PAD} extra padding')
        return cls(alphabet, max_len=max_len + EXTRA_PAD, *args, **kwargs)

    @classmethod
    def smiles_to_selfies_list(cls, smiles):
        selfies_str = str(selfies.encoder(smiles, PrintErrorMessage=True))
        selfies_list = selfies_str.replace('[', '').split(']')[:-1]
        return selfies_list

    @classmethod
    def selfies_list_to_smiles(cls, selfies_list, pad_char=SS.pad):
        selfies_str = '[' + ']['.join(selfies_list) + \
                      ']'.replace(f'[{pad_char}]', '')
        smiles = str(selfies.decoder(selfies_str, PrintErrorMessage=False))
        return smiles

    def string_preprocess(self, astr):
        return self.smiles_to_selfies_list(astr)

    def charlist_postprocess(self, char_list):
        selfies_str = self.selfies_list_to_smiles(char_list)
        return 'Error:' + ''.join(char_list) if selfies_str in [-1, '-1'] else isosmiles(selfies_str)


class MOFVocab:
    @classmethod
    def from_data(cls, df, columns, weighting=True, *args, **kwargs):

        encoders = OrderedDict()
        weights = OrderedDict()
        for col in columns:
            enc = LabelEncoder()
            enc.fit(df[col].values)
            values = df[col].tolist()
            mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
            labels, counts = np.unique(values, return_counts=True)
            max_count = np.max(counts)
            w = np.zeros(len(enc.classes_), dtype=np.float32)
            for label, count in zip(labels, counts):
                if weighting:
                    w[mapping[label]] = max_count / count
                else:
                    w[mapping[label]] = 1.0

            weights[col] = w
            encoders[col] = enc
        print(f'Used columns ={columns} with frequency weighting={weighting}')
        for col in columns:
            print(f'{col:12s} has {len(encoders[col].classes_)} classes')
        return cls(encoders, weights, *args, **kwargs)

    def __init__(self, encoders, weights):

        self.categories = list(encoders.keys())
        self.weights = weights
        self.weight_list = [w for w in weights.values()]
        self.encoders = encoders
        self.n_encoders = len(encoders)
        self.dims = [len(enc.classes_) for enc in encoders.values()]
        self.total_dim = sum(self.dims)

    def __len__(self):
        return sum(self.dims)

    def get_label_to_id(self, key):
        enc = self.encoders[key]
        return dict(zip(enc.classes_, range(len(enc.classes_))))

    def get_id2label(self, key):
        enc = self.encoders[key]
        return dict(zip(range(len(enc.classes_)), enc.classes_))

    def df_to_ids(self, df, batch=10000):
        n = len(df)
        mof_ids = np.zeros((n, self.n_encoders), dtype=np.int)
        n_loops = int(np.ceil(len(df) / batch))
        disable = n_loops < 5
        for indexes in tqdm(chunked(range(n), batch), total=n_loops, disable=disable, desc='MOF'):
            sub_df = df.iloc[list(indexes)]
            for index, (col, enc) in enumerate(self.encoders.items()):
                mof_ids[indexes, index] = enc.transform(sub_df[col].values)
        return mof_ids

    def mof_to_ids(self, mof):
        ids = []
        for i, enc in enumerate(self.encoders.values()):
            arr = enc.transform([mof[i]])[0]
            ids.append(arr)
        return ids

    def ids_to_mof(self, ids):
        mof = []
        for i, enc in enumerate(self.encoders.values()):
            cat = enc.inverse_transform([ids[i]])[0]
            mof.append(cat)

        return mof

    def ids_array_to_mof_list(self, ids_arr):
        cats = []
        for i, enc in enumerate(self.encoders.values()):
            cats.append(enc.inverse_transform(ids_arr[:, i]))
        return np.array(cats).T.tolist()


class PropVocab:

    @classmethod
    def from_data(cls, df: pd.DataFrame,
                  labels: List[Text],
                  weights: Optional[np.ndarray] = None,
                  scaler_type:Optional[Text]=None):
        if scaler_type is None:
            scaler = StandardScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer()
        elif scaler_type == 'quantile':
            scaler = QuantileTransformer()
        else:
            raise ValueError(f'{scaler_type} not implemented!')
        return cls(scaler.fit(df[labels].values), labels, weights)

    def __init__(self, scaler: Any, labels: List[Text], weights: Optional[np.ndarray] = None):
        self.scaler = scaler
        self.labels = labels
        if weights is None:
            weights = np.ones(len(labels)).astype(np.float32)
        self.weights = weights

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(x)

    def df_to_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.scaler.transform(df[self.labels]), np.ones(len(df), dtype=np.float32)

    def invalid_values(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        invalid_value = -5000 * np.ones_like(self.scaler.mean_)
        return np.array([invalid_value] * n, dtype=np.float32), np.zeros(n, dtype=np.float32)


if __name__ == '__main__':

    TEST_DATA = 'data/qm9_prop.csv'
    df = pd.read_csv(TEST_DATA).iloc[:10000]
    smiles_list = df['isosmiles'].tolist()

    for vocab_class in [CharVocab, SELFIESVocab]:
        print(f'Testing {vocab_class.__name__}..')
        vocab = vocab_class.from_data(smiles_list)
        ids_list = [vocab.string_to_ids(smi) for smi in smiles_list]
        new_smiles = [vocab.ids_to_string(ids) for ids in ids_list]
        same = [smi == isosmiles(new_smi) for smi, new_smi in zip(smiles_list, new_smiles)]
        if not all(same):
            index = np.argmin(same)
            sampler = f'{smiles_list[index]} and {new_smiles[index]}'
            raise ValueError(f"{vocab_class.__name__} does not decode perfectly, e.g. {sampler}")

    vocab_class = MOFVocab
    print(f'Testing {vocab_class.__name__}..')
    MOF_DATA = 'data/MOF_gen_train.csv'
    sample_df = pd.read_csv(MOF_DATA)
    mof_columns = ['metal_node', 'organic_core', 'topology', 'id2mof']
    vocab = vocab_class.from_data(sample_df, mof_columns, weighting=True)
    inputs = sample_df[mof_columns].values.tolist()
    n = len(inputs)
    ids = vocab.df_to_ids(sample_df)
    outputs = vocab.ids_array_to_mof_list(ids)
    sames = [inputs[i] == outputs[i] for i in range(n)]
    if not all(same):
        index = np.argmin(same)
        sampler = f'{inputs[index]} and {outputs[index]}'
        raise ValueError(f"{vocab_class.__name__} does not decode perfectly, e.g. {sampler}")
