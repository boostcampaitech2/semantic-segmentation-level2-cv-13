import torch
import random
import numpy as np
import argparse


def arg_parse():
    """
    parse arguments from a command
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    return args


def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

def _fast_hist(label_true, label_pred, n_class):
    """
    make confusion matrix for a single image
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                       minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [mean_iu]: mean IU
    """
    diag_hist = torch.diag(hist)

    acc = diag_hist.sum() / hist.sum()

    iu = diag_hist / (hist.sum(axis=1) + hist.sum(axis=0) - diag_hist)
    mean_iu = torch.div(torch.nansum(iu, dim=0),
                        (~torch.isnan(iu)).count_nonzero(dim=0))

    return acc, mean_iu, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask].int() + label_pred[mask],
        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
