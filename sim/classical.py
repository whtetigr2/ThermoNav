"""Classical baseline: independent-voxel log-odds map (OctoMap-style, no pairwise prior)."""
import numpy as np


def classical_marginals(logodds):
    """P(x_v=1) = sigmoid(L_v), independent per voxel."""
    return 1.0 / (1.0 + np.exp(-np.clip(logodds, -30, 30)))


def accuracy(pred_prob, truth, mask=None):
    """Fraction of voxels where (prob > 0.5) == truth."""
    pred = (pred_prob > 0.5).astype(np.uint8)
    if mask is not None:
        pred = pred[mask]
        truth = truth[mask]
    return float(np.mean(pred == truth))


def observed_mask(logodds, thresh=0.05):
    """Voxels with |log-odds| above thresh (sensor actually saw them)."""
    return np.abs(logodds) > thresh
