import os
from typing import List, Text, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from vaemof import configs
from vaemof.configs import AttributeDict
from vaemof.losses import masked_wmse_loss
from vaemof.modules import make_mlp
from vaemof.vocabs import MOFVocab, OneHotVocab, PropVocab, SELFIESVocab

tensor = torch.tensor
DataTuple = Tuple[Text, Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]


def reparameterize(mu: tensor, log_var: tensor):
    """Reparametrization trick."""
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(mu)
    kl_loss = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(1).mean()
    z = mu + eps * std
    return z, kl_loss


class MOFEncoder(nn.Module):
    """Mof encoder for a VAE."""

    def __init__(self, latent_dim: int, mof_dims: List[int], n_layers: int, act: Text,
                 batchnorm: bool, dropout: float):
        super().__init__()
        self.feat_embs = nn.ModuleList([nn.Embedding(n, latent_dim) for n in mof_dims])
        self.n_features = len(mof_dims)
        self.mof_dims = mof_dims
        self.mlp = make_mlp([latent_dim] * n_layers + [latent_dim * 2], act, batchnorm, dropout, activation_last=True)

    def forward(self, mof: tensor) -> tensor:
        h = []
        for index, emb in enumerate(self.feat_embs):
            h.append(emb(mof[:, index]))
        h = torch.sum(torch.stack(h), dim=0)
        h = self.mlp(h)
        return h


class MOFDecoder(nn.Module):
    """A mof decoder for a VAE."""

    def __init__(self, latent_dim: int, vocab_mof: MOFVocab, wloss: bool,
                 n_layers: int, act: Text, batchnorm: bool, dropout: float):
        super().__init__()
        self.vocab_mof = vocab_mof
        self.wloss = wloss
        self.n_features = len(vocab_mof.dims)
        self.mof_dims = vocab_mof.dims
        weights = [torch.from_numpy(w.astype(np.float32)) for w in vocab_mof.weight_list]
        self.mof_weights = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w in weights])
        self.mlp = make_mlp([latent_dim] * (n_layers + 1), act, batchnorm, dropout, activation_last=True)
        self.out_to_id = nn.ModuleList([nn.Linear(latent_dim, i) for i in vocab_mof.dims])

    def forward(self, mof: tensor, z: tensor) -> Tuple[tensor, tensor]:
        out = self.mlp(z)
        mof_hat = []
        loss = 0.0
        for index, cat_map in enumerate(self.out_to_id):
            w = self.mof_weights[index] if self.wloss else None
            target = mof[:, index]
            output = cat_map(out)
            mof_hat.append(torch.argmax(F.softmax(output, dim=-1), dim=1))
            loss += F.cross_entropy(output, target, weight=w)
        mof_hat = torch.transpose(torch.stack(mof_hat), 0, 1)
        return loss, mof_hat

    def z_to_mof(self, z: tensor):
        """Latent vector to mof."""
        with torch.no_grad():
            out = self.mlp(z)
            mof_ids = []
            for index, cat_map in enumerate(self.out_to_id):
                output = cat_map(out)
                ids = torch.argmax(F.softmax(output, dim=-1), dim=1)
                mof_ids.append(ids.detach().cpu().numpy())
            mof_ids_list = np.stack(mof_ids).T.tolist()
        mofs = [self.vocab_mof.ids_to_mof(mof) for mof in mof_ids_list]
        return mofs


class PropDecoder(nn.Module):
    """Property predictor based on latent space."""

    def __init__(self, latent_dim: int, weights: List[float], scaler: Any,
                 n_layers: int, act: Text, batchnorm: bool, dropout: float):
        super().__init__()
        output_dim = len(weights)
        self.scaler = scaler
        weights = torch.from_numpy(np.array(weights).astype(np.float32))
        self.loss_weights = nn.Parameter(weights, requires_grad=False)
        layer_dims = [latent_dim] * n_layers + [output_dim]
        self.mlp = make_mlp(layer_dims, act, batchnorm, dropout, activation_last=False)

    def forward(self, z: tensor, y: tensor, mask: tensor) -> Tuple[tensor, tensor]:
        y_hat = self.mlp(z)
        loss = masked_wmse_loss(y_hat, y, mask, self.loss_weights)
        return loss, y_hat

    def z_to_y(self, z: tensor) -> np.ndarray:
        with torch.no_grad():
            y_hat = self.mlp(z).numpy()
        y_hat = self.scaler.inverse_transform(y_hat)
        return y_hat


class CharEncoder(nn.Module):
    """Char rnn based encoder for a VAE."""

    def __init__(self, vocab, latent_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        n_vocab, enc_emb = len(vocab), vocab.vectors.size(1)
        self.emb = nn.Embedding(n_vocab, enc_emb, vocab.pad)
        self.emb.weight.data.copy_(vocab.vectors)
        self.rnn = nn.GRU(
            enc_emb,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False)
        self.linear = nn.Linear(hidden_dim * n_layers, latent_dim * 2)

    def forward(self, x: tensor) -> tensor:
        x_emb = [self.emb(i_x) for i_x in x]
        x_pack = nn.utils.rnn.pack_sequence(x_emb)
        _, h = self.rnn(x_pack)
        n_layers, batch_size, hidden_dim = h.shape
        # [n_layers, batch_size, hidden_dim] -> [batch_size,hidden_dim*n_layers]
        h = h.view(batch_size, hidden_dim * n_layers)
        out = self.linear(h)
        return out


class CharDecoder(nn.Module):
    """Char rnn based decoder for a VAE."""

    def __init__(self, vocab, emb, latent_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        n_vocab, enc_emb = len(vocab), vocab.vectors.size(1)
        self.vocab = vocab
        self.emb = emb
        self.rnn = nn.GRU(
            enc_emb + latent_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.z_to_h = nn.Linear(latent_dim, hidden_dim)
        self.h_to_char = nn.Linear(hidden_dim, n_vocab)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: tensor, z: tensor) -> Tuple[tensor, tensor]:
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.vocab.pad)
        x_emb = self.emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.z_to_h(z)
        h_0 = h_0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)

        output, _ = self.rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.h_to_char(output)
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.vocab.pad
        )

        x_hat = torch.argmax(F.softmax(y, dim=-1), dim=-1)
        return recon_loss, x_hat

    def z_to_x(self, z: tensor, max_len: Optional[int] = None, temp: float = 1.0):
        """Latent vector to selfies."""
        device = self.device
        bos, pad, eos = self.vocab.bos, self.vocab.pad, self.vocab.eos
        max_len = max_len or self.vocab.max_len
        with torch.no_grad():
            n_batch = z.shape[0]
            z_0 = z.unsqueeze(1)
            # Initial values
            h = self.z_to_h(z)
            h = h.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
            w = torch.tensor(bos, device=device).repeat(n_batch)
            x = torch.tensor([pad], device=device).repeat(n_batch, max_len)
            x[:, 0] = self.vocab.bos
            end_pads = torch.tensor([max_len], device=device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=device)
            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)
                o, h = self.rnn(x_input, h)
                y = self.h_to_char(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)
                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & torch.eq(w, eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]].numpy())
        new_x = [self.vocab.ids_to_string(i) for i in new_x]
        return new_x


class VAEMOF(nn.Module):
    def __init__(self, hparams: AttributeDict, vocab: OneHotVocab, vocab_mof: Optional[MOFVocab] = None,
                 vocab_y: Optional[PropVocab] = None):
        super().__init__()

        # Setup variables.
        self.vocab = vocab
        self.vocab_mof = vocab_mof
        self.vocab_y = vocab_y
        self.hparams = hparams
        self.use_decoder = hparams.vae_selfies_dec
        self.use_mof_decoder = hparams.vae_mof_dec
        self.use_mof_encoder = hparams.vae_mof_enc
        self.use_y_decoder = hparams.vae_y_dec
        self.use_kl = self.use_mof_decoder or self.use_decoder

        latent_dim = hparams.vae_latent_dim

        # Latent space mappings.
        self.z_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.z_logvar = nn.Linear(latent_dim * 2, latent_dim)

        self.enc_x = CharEncoder(vocab=vocab, latent_dim=latent_dim,
                                 hidden_dim=hparams.enc_hidden_dim,
                                 n_layers=hparams.enc_n_layers,
                                 dropout=hparams.enc_dropout)
        if self.use_decoder:
            self.dec_x = CharDecoder(vocab=vocab, emb=self.enc_x.emb,
                                     latent_dim=latent_dim,
                                     hidden_dim=hparams.dec_hidden_dim,
                                     n_layers=hparams.dec_n_layers,
                                     dropout=hparams.dec_dropout)
        if self.use_mof_encoder:
            self.enc_mof = MOFEncoder(latent_dim=latent_dim,
                                      mof_dims=vocab_mof.dims,
                                      n_layers=2, act='relu',
                                      batchnorm=False, dropout=0.0)
        if self.use_mof_decoder:
            self.dec_mof = MOFDecoder(latent_dim=latent_dim,
                                      vocab_mof=vocab_mof,
                                      wloss=hparams.mof_weighted_loss,
                                      n_layers=1, act='relu',
                                      batchnorm=False, dropout=0.0)
        if self.use_y_decoder:
            self.dec_y = PropDecoder(latent_dim=latent_dim,
                                     weights=vocab_y.weights,
                                     scaler=vocab_y.scaler,
                                     n_layers=1, act='relu',
                                     batchnorm=False, dropout=0.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, as_snapshot=False):
        at_results_dir = lambda x: os.path.join(self.hparams.files_results, x)
        configs.check_files(self.hparams)
        device = self.device
        self = self.to('cpu')
        if as_snapshot:
            torch.save(self.state_dict(), at_results_dir('snapshot'))
        else:
            self.hparams.save_to_json(at_results_dir(self.hparams.files_config))
            torch.save(self.state_dict(), at_results_dir('model'))
            torch.save(self.vocab, at_results_dir('vocab'))
            if self.vocab_mof:
                torch.save(self.vocab_mof, at_results_dir('vocab_mof'))
            if self.vocab_y:
                torch.save(self.vocab_y, at_results_dir('vocab_y'))
        self = self.to(device)

    @classmethod
    def load(cls, hparams):
        at_results_dir = lambda x: os.path.join(hparams.files_results, x)
        if not os.path.exists(hparams.files_results):
            raise ValueError(f'results_dir={hparams.files_results} does not exists')
        if os.path.exists(at_results_dir('vocab_y')):
            vocab_y = torch.load(at_results_dir('vocab_y'))
        else:
            vocab_y = None
        if os.path.exists(at_results_dir('vocab_mof')):
            vocab_mof = torch.load(at_results_dir('vocab_mof'))
        else:
            vocab_mof = None
        vocab = torch.load(at_results_dir('vocab'))
        model = cls(hparams, vocab, vocab_mof, vocab_y)
        model_state = torch.load(at_results_dir('model'))
        model.load_state_dict(model_state)
        return model

    def df_to_tuples(self, df: pd.DataFrame, smiles_column: Text) -> List[DataTuple]:
        n = len(df)
        smi_ids = self.vocab.df_to_ids(df, smiles_column)
        mof_ids = self.vocab_mof.df_to_ids(df)
        has_y = all(i in df.columns.tolist() for i in self.vocab_y.labels)
        y, y_mask = self.vocab_y.df_to_y(df) if has_y else self.vocab_y.invalid_values(n)
        return list(zip(*[smi_ids, mof_ids, y, y_mask]))

    def tuples_to_tensors(self, data: List[DataTuple]):
        device = self.device
        outs = {'x': None, 'mof': None, 'y': None, 'y_mask': None}
        data.sort(key=lambda x: len(x[0]), reverse=True)
        outs['x'] = [torch.tensor(x[0], dtype=torch.long, device=device) for x in data]

        if self.use_mof_encoder:
            mof_arr = np.vstack([t[1] for t in data]).astype(np.int)
            outs['mof'] = torch.from_numpy(mof_arr).to(device)

        if self.use_y_decoder:
            y_arr = np.vstack([t[2] for t in data]).astype(np.float32)
            outs['y'] = torch.from_numpy(y_arr).to(device)
            y_mask = np.vstack([t[3] for t in data]).astype(np.float32)
            outs['y_mask'] = torch.from_numpy(y_mask).to(device)

        return outs

    def check_inputs(self, mof: Optional[tensor] = None, y: Optional[tensor] = None, y_mask: Optional[tensor] = None):
        if mof is None and (self.use_mof_encoder or self.use_mof_decoder):
            raise ValueError('use_mof_encoder and mof is missing')
        if (y is None or y_mask is None) and self.use_y_decoder:
            raise ValueError('use_y_decoder=True, missing y or y_mask!')

    def forward(self, x: tensor, mof: Optional[tensor] = None,
                y: Optional[tensor] = None, y_mask: Optional[tensor] = None):
        """Forward pass for MOFVAE"""
        self.check_inputs(mof, y, y_mask)
        outs = {'x': None, 'mof': None, 'y': None}
        losses = {'kl': 0.0, 'x': 0.0, 'mof': 0.0, 'y': 0.0}
        h_x = self.enc_x(x)
        h_mof = self.enc_mof(mof) if self.use_mof_encoder else 0.0
        h = h_x + h_mof
        z, kl = reparameterize(self.z_mu(h), self.z_logvar(h))
        losses['kl'] = kl if self.use_kl else 0.0
        if self.use_decoder:
            losses['x'], outs['x'] = self.dec_x(x, z)

        if self.use_mof_decoder:
            losses['mof'], outs['mof'] = self.dec_mof(mof, z)

        if self.use_y_decoder:
            losses['y'], outs['y'] = self.dec_y(z, y, y_mask)

        return losses, outs

    def sample_z_prior(self, n_batch: int) -> tensor:
        """Sampling z ~ p(z) = N(0, I)."""
        return torch.randn(n_batch, self.z_mu.out_features,
                           device=self.device)

    def inputs_to_z(self, x: tensor, mof: Optional[tensor] = None) -> tensor:
        with torch.no_grad():
            h_x = self.enc_x(x)
            h_mof = self.enc_mof(mof) if self.use_mof_encoder else 0.0
            h = h_x + h_mof
            z, _ = reparameterize(self.z_mu(h), self.z_logvar(h))
        return z

    def z_to_outputs(self, n_batch: int, z: Optional[tensor] = None, max_len: Optional[int] = None, temp: float = 1.0):
        """Generating n_batch samples in eval mode."""
        outs = {}
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            if self.use_decoder:
                outs['x'] = self.dec_x.z_to_x(z, max_len, temp)
            if self.use_y_decoder:
                outs['y'] = self.dec_y.z_to_y(z)
            if self.use_mof_decoder:
                outs['mof'] = self.dec_mof.z_to_mof(z)
        return outs


if __name__ == '__main__':
    from vaemof import configs
    from vaemof import utils
    from vaemof import modules
    from vaemof import training

    MOF_DATA = 'data/MOF_properties_train.csv'
    sample_df = pd.read_csv(MOF_DATA)
    smiles_column = 'branch_smiles'
    vocab = SELFIESVocab.from_data(sample_df[smiles_column].tolist())
    mof_columns = ['metal_node', 'organic_core', 'topology']
    vocab_mof = MOFVocab.from_data(sample_df, mof_columns, weighting=True)
    y_columns = ['lcd', 'pld', 'density']
    vocab_y = PropVocab.from_data(sample_df, y_columns)
    torch.cuda.empty_cache()
    for preset in configs.PRESETS:
        utils.header_str(preset)
        hparams = configs.get_model_config(preset)
        model = VAEMOF(hparams, vocab, vocab_mof, vocab_y)
        modules.model_summary(model, include_children=False)
        data = model.df_to_tuples(sample_df, smiles_column)
        data_loader = training.get_dataloader(data, model, True)
        batch = next(iter(data_loader))
        losses, outputs = model(batch['x'], batch['mof'], batch['y'], batch['y_mask'])
        out = model.z_to_outputs(4)