import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys; sys.path.insert(0, '.')
import time
import numpy as np, jax.numpy as jnp
from train_s2_theta_v2 import train, seq_to_bool, N
rng = np.random.default_rng(0)
M = 128
h_dyn = rng.normal(0, 0.25, (M, N)).astype(np.float32)
seq = rng.integers(0, 6, (M, 3))
d = seq_to_bool(seq)
print('data check: each timestep one-hot?', [(d[:, t*6:(t+1)*6].sum(axis=1) == 1).all() for t in range(3)])
t0=time.time()
b, w, losses = train('random', h_dyn, seq, n_epochs=2, batch=32, seed=0)
print('train 2 epochs OK in %.0fs' % (time.time()-t0))
print('finite:', bool(np.isfinite(b).all() and np.isfinite(w).all()))
print('max|b|=%.2f max|w|=%.2f' % (np.abs(b).max(), np.abs(w).max()))
print('losses:', np.round(losses, 4).tolist())
