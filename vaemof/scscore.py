'''
This is a standalone, importable SCScorer model. It does not have tensorflow as a
dependency and is a more attractive option for deployment. The calculations are
fast enough that there is no real reason to use GPUs (via tf) instead of CPUs (via np)
'''

import gzip
import json
import math
import os

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import six

WEIGHTS_FILE = 'data/scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz'

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SCScorer():
    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False

    def restore(self, weight_path=WEIGHTS_FILE, FP_rad=FP_rad, FP_len=FP_len):
        self.FP_len = FP_len;
        self.FP_rad = FP_rad
        self._load_vars(weight_path)
        print('Restored variables from {}'.format(weight_path))
        self._restored = True
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(Chem.MolFromSmiles(smi))

    def mol_to_fp(self, mol):
        if mol is None:
            return np.array((self.FP_len,), dtype=np.uint8)
        fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True)  # uitnsparsevect
        fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
        for k, v in six.iteritems(fp.GetNonzeroElements()):
            fp_folded[k % self.FP_len] += v
        return np.array(fp_folded)

    def apply(self, x):
        if not self._restored:
            raise ValueError('Must restore model weights!')
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars) - 2)
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v: print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return smi, cur_score

    def _load_vars(self, weight_path):
        with gzip.GzipFile(weight_path, 'r') as fin:  # 4. gzip
            json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)
            json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
            self.vars = json.loads(json_str)
            self.vars = [np.array(x) for x in self.vars]


if __name__ == '__main__':

    model = SCScorer()
    model.restore(WEIGHTS_FILE)
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))
