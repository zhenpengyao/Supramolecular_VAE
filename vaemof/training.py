from collections import OrderedDict, defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vaemof.configs import at_results_dir
from vaemof.vocabs import valid_smiles

OUTPUTS = ['x', 'mof', 'y', 'y_mask']
COMPONENTS = ['kl', 'x', 'mof', 'y']
ALL_COMPONENTS = COMPONENTS + ['loss']


class CyclicScheduler:

    def __init__(self, cycle_length, const_length, n_epochs, w_start=0.0, w_end=1.0):
        self.cycle_length = cycle_length
        self.const_length = const_length
        self.n_epochs = n_epochs
        n_cycles = np.floor(n_epochs / (cycle_length + const_length))
        self.max_cycle_epoch = n_cycles * (cycle_length + const_length)
        self.w_start = w_start
        self.w_end = w_end

    def __call__(self, i):
        cur_epoch = i % (self.cycle_length + self.const_length)
        growing = cur_epoch < self.cycle_length and i < self.max_cycle_epoch
        if growing:
            rate = (self.w_end - self.w_start)
            t = cur_epoch / self.cycle_length
            return rate * t + self.w_start
        else:
            return self.w_end


class ConstScheduler:

    def __init__(self, value):
        self.value = value

    def __call__(self, i):
        return self.value


class LinearScheduler:

    def __init__(self, start, end, w_start=0.0, w_end=1.0):
        self.i_start = start
        self.i_end = end
        self.w_start = w_start
        self.w_end = w_end
        self.rate = (self.w_end - self.w_start) / (self.i_end - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        w = min(self.w_start + k * self.rate, self.w_end)
        return w


def plot_scheduler(scheduler, n_epochs: int, label=''):
    epochs = np.arange(n_epochs)
    values = [scheduler(i) for i in epochs]
    plt.plot(epochs, values, label=label)


def get_dataloader(model, data, training: bool = True):
    return DataLoader(data, batch_size=model.hparams.train_batch_size,
                      shuffle=training, collate_fn=model.tuples_to_tensors, drop_last=training)


class TrainStats:

    def __init__(self, filename):
        self.filename = filename
        self.stats = None
        self.report_stats = None
        self.trues = None
        self.preds = None
        self.results = []
        self.buffers = None

    @property
    def report(self):
        return self.report_stats

    def setup_batch_buffers(self, train_n, test_n):
        buffers = OrderedDict()
        for key in ALL_COMPONENTS:
            buffers[f'train_{key}'] = deque(maxlen=train_n)
            buffers[f'test_{key}'] = deque(maxlen=test_n)
        self.buffers = buffers

    def start_epoch(self, epoch, lr, weights):
        self.stats = OrderedDict([('epoch', epoch)])
        self.report_stats = OrderedDict([('lr', lr)])
        self.report_stats.update([(f'λ_{key}', v) for key, v in weights.items()])
        self.preds = defaultdict(list)
        self.trues = defaultdict(list)

    def update_batch(self, prefix, losses, trues=None, preds=None):
        reported = {}
        for key in ALL_COMPONENTS:
            value = losses[key].item()
            self.buffers[f'{prefix}_{key}'].append(value)
            reported[key] = np.mean(self.buffers[f'{prefix}_{key}'])
        if trues and preds:
            for key in preds:
                if preds[key] is not None:
                    self.preds[key].append(preds[key].numpy())
            for key in trues:
                if trues.get(key) is not None:
                    self.trues[key].append(trues[key].numpy())

        return reported

    def update_epoch(self, prefix, reportable=False):
        for key in ALL_COMPONENTS:
            full_key = f'{prefix}_{key}'
            if reportable:
                self.report_stats[full_key] = np.mean(self.buffers[full_key])
            else:
                self.stats[full_key] = np.mean(self.buffers[full_key])

    def compute_metrics(self, model):
        def flat_map(x, inner_fn=lambda x: x, outer_fn=lambda x: x):
            return outer_fn([inner_fn(j) for i in x for j in i])

        if self.preds['x']:
            recon_smiles = flat_map(self.preds['x'], model.vocab.ids_to_string)
            valid = np.array(list(map(valid_smiles, recon_smiles)))
            self.report_stats['valid_smiles'] = np.sum(valid) / len(valid) * 100.0
        if self.preds['mof']:
            mof_hat = flat_map(self.preds['mof'], outer_fn=np.stack)
            mof = flat_map(self.trues['mof'], outer_fn=np.stack)
            samesies = np.sum(np.equal(mof, mof_hat))
            self.report_stats['mof_acc'] = samesies / np.prod(mof.shape) * 100.0
        if self.preds['y']:
            y_true = flat_map(self.trues['y'], outer_fn=np.stack)
            y_pred = flat_map(self.preds['y'], outer_fn=np.stack)
            mask = flat_map(self.trues['y_mask'], outer_fn=np.stack)
            mask = mask.ravel().astype(bool)
            values = OrderedDict()
            labels = model.vocab_y.labels
            for i, label in enumerate(labels):
                values[label] = sklearn.metrics.r2_score(y_true[mask, i], y_pred[mask, i])
            self.report_stats['mean_r2'] = np.nanmean(list(values.values()))
            self.stats.update(values)

    def finalize_epoch(self, save=True):
        self.stats.update(self.report_stats)
        self.results.append(self.stats)
        if save:
            pd.DataFrame(self.results).to_csv(self.filename, index=None)


class Trainer:

    def __init__(self, hparams):
        self.hparams = hparams
        self.stats = TrainStats(at_results_dir(hparams, 'files_log'))
        self.schedulers = self.setup_schedulers(hparams)
        self.n_epochs = hparams.train_epochs

    def setup_schedulers(self, hparams):
        schedulers = OrderedDict()
        n_epochs = hparams.train_epochs
        schedulers['x'] = ConstScheduler(1.0)
        schedulers['kl'] = CyclicScheduler(cycle_length=hparams.kl_cycle_length,
                                           const_length=hparams.kl_cycle_constant,
                                           n_epochs=n_epochs,
                                           w_start=hparams.kl_weight_start,
                                           w_end=hparams.kl_weight_end)
        schedulers['y'] = LinearScheduler(start=hparams.y_start,
                                          end=n_epochs - hparams.y_const_length,
                                          w_start=hparams.y_w_start,
                                          w_end=hparams.y_w_end)
        schedulers['mof'] = LinearScheduler(start=hparams.mof_start,
                                            end=n_epochs - hparams.mof_const_length,
                                            w_start=hparams.mof_w_start,
                                            w_end=hparams.mof_w_end)
        return schedulers

    def train(self, model, train_data, test_data):
        lr = self.hparams.train_lr
        train_loader = get_dataloader(model, train_data, True)
        test_loader = get_dataloader(model, test_data, False)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        self.stats.setup_batch_buffers(len(train_loader), len(test_loader))
        pbar = tqdm(range(self.hparams.train_epochs), desc='Epochs')
        for epoch in pbar:
            weights = {key: sch(epoch) for key, sch in self.schedulers.items()}
            self.stats.start_epoch(epoch, lr, weights)
            self.train_step(model, train_loader, optimizer, weights)
            self.test_step(model, test_loader, weights)
            pbar.set_postfix(self.stats.report)
            self.stats.finalize_epoch()

        model.save()

    def train_step(self, model, data_loader, optimizer, weights):
        model.train()
        pbar = tqdm(data_loader, leave=False, desc='Train')
        for batch in pbar:
            losses, _ = model(batch['x'], batch['mof'], batch['y'], batch['y_mask'])
            w_loss = {k: weights[k] * losses[k] for k in COMPONENTS}
            loss = w_loss['x'] + w_loss['kl'] + w_loss['mof'] + w_loss['y']
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), self.hparams.train_clip_grad)
            optimizer.step()
            losses['loss'] = loss
            pbar.set_postfix(self.stats.update_batch('train', losses))
        self.stats.update_epoch('train', reportable=True)

    @torch.no_grad()
    def test_step(self, model, data_loader, weights):
        model.eval()
        pbar = tqdm(data_loader, leave=False, desc='Test')
        for batch in pbar:
            losses, preds = model(batch['x'], batch['mof'], batch['y'], batch['y_mask'])
            w_loss = {k: weights[k] * losses[k] for k in COMPONENTS}
            loss = w_loss['x'] + w_loss['kl'] + w_loss['mof'] + w_loss['y']
            losses['loss'] = loss
            trues = {key: batch[key] for key in ['mof', 'y', 'y_mask']}
            self.stats.update_batch('test', losses, trues, preds)

        self.stats.update_epoch('test', reportable=False)
        self.stats.compute_metrics(model)
