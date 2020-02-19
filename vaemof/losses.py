import numpy as np
import torch
import torch.nn.functional as F


def masked_mse_loss(y_hat, y, mask):
    loss_tensor = F.mse_loss(y_hat * mask, y * mask, reduction='none')
    loss = torch.sum(loss_tensor) / (torch.sum(mask) + 1e-7)
    return loss


def masked_wmse_loss(y_hat, y, mask, w):
    loss_tensor = F.mse_loss(y_hat * mask, y * mask, reduction='none') * w
    loss = torch.sum(loss_tensor) / (torch.sum(mask) + 1e-7)
    return loss


def test_masked_loss(data):
    extra = 1.0
    y_true = torch.Tensor(np.array([t[2] for t in test_data]))
    mask = torch.Tensor(np.array([t[3] for t in test_data]).astype(np.float32))
    n_dim = y_true.shape[-1]
    loss = masked_mse_loss(y_true, y_true, mask).item()
    assert loss == 0.0, "masked_loss(y,y): should be 0, it's {}".format(loss)
    y_pred = torch.Tensor(
        np.array([t[2] if t[3][0] else t[2] + extra for t in test_data]))
    loss = masked_mse_loss(y_true, y_pred, mask).item()
    assert loss == 0.0, "masked_loss(y,y+1[False]): should be 0, it's {}".format(
        loss)
    y_pred = torch.Tensor(
        np.array([t[2] + extra if t[3][0] else t[2] for t in test_data]))
    loss = masked_mse_loss(y_true, y_pred, mask).item()
    correct_loss = extra * n_dim
    assert loss == correct_loss, "masked_loss(y,y+1[True]): should be {}, it's {}".format(
        correct_loss, loss)
    return True


def test_masked_wloss(data):
    extra = 1.0

    y_true = torch.Tensor(np.array([t[2] for t in test_data]))
    mask = torch.Tensor(np.array([t[3] for t in test_data]).astype(np.float32))
    n_dim = y_true.shape[-1]
    w = torch.Tensor(np.ones(n_dim))
    loss = masked_wmse_loss(y_true, y_true, mask, w).item()
    assert loss == 0.0, "masked_loss(y,y): should be 0, it's {}".format(loss)
    y_pred = torch.Tensor(
        np.array([t[2] if t[3][0] else t[2] + extra for t in test_data]))
    loss = masked_wmse_loss(y_true, y_pred, mask, w).item()
    assert loss == 0.0, "masked_loss(y,y+1[False]): should be 0, it's {}".format(
        loss)
    y_pred = torch.Tensor(
        np.array([t[2] + extra if t[3][0] else t[2] for t in test_data]))
    loss = masked_wmse_loss(y_true, y_pred, mask, w).item()
    correct_loss = extra * n_dim
    assert loss == correct_loss, "masked_loss(y,y+1[True]): should be {}, it's {}".format(
        correct_loss, loss)
    w = np.random.random(n_dim)
    w_mean = np.mean(w)
    w = torch.Tensor(w)
    y_pred = torch.Tensor(
        np.array([t[2] + extra if t[3][0] else t[2] for t in test_data]))
    loss = masked_wmse_loss(y_true, y_pred, mask, w).item()
    correct_loss = extra * n_dim * w_mean
    assert np.isclose(
        loss, correct_loss), "masked_loss(y,y+1[True]): should be {}, it's {}".format(correct_loss, loss)
    return True
