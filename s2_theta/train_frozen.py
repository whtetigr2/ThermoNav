"""Train with temporal couplings frozen at zero (63 learnable params)."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from thrml import Block, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init
sys.path.insert(0, '.')
from model_def import build_nodes_edges, N, VIS_N, K

res = build_nodes_edges()
nodes, eidx = res[0], res[2]
vis_nodes, free_blocks, clamped_blocks = res[3], res[4], res[5]
h_eng, J_eng = res[6], res[7]
beta = jnp.array(1.0)
SCHED = SamplingSchedule(8, 16, 2)
Z1_MAX = 6.0

# Freeze temporal at zero: only learn one-hot block weights
def is_temporal(i, j):
    return abs(i // K - j // K) == 1
learn_mask = np.array([not is_temporal(i, j) for (i, j) in eidx])  # 45 True
learn_idx = np.where(learn_mask)[0]
print(f"Learnable weights: {len(learn_idx)}/105, total params: {N + len(learn_idx)}")

# Build edge list with only learnable edges for the model
# Actually, we need ALL edges in the model (frozen at 0), but only update learnable.
# Simpler: keep all 105 weights, but zero-out gradients for frozen ones.
w_eng_full = np.array([J_eng[i, j] for (i, j) in eidx], dtype=np.float32)
b_eng = h_eng.astype(np.float32)

def single_grad(biases, weights, h_dyn, data, key):
    model = IsingEBM(nodes, res[1], biases + h_dyn, weights, beta)
    spec = IsingTrainingSpec(model, [Block(vis_nodes)], [],
                             clamped_blocks, free_blocks, SCHED, SCHED)
    k1, k2, k3 = jax.random.split(key, 3)
    init_pos = hinton_init(k1, model, clamped_blocks, (1, 1))
    init_neg = hinton_init(k2, model, free_blocks, (1,))
    gw, gb, _, _ = estimate_kl_grad(k3, spec, nodes, res[1], [data[None, :]], [],
                                    init_pos, init_neg)
    return gb, gw

@jax.jit
def batch_grads(biases, weights, h_dyn_b, data_b, keys):
    gb, gw = jax.vmap(single_grad, in_axes=(None, None, 0, 0, 0))(
        biases, weights, h_dyn_b, data_b, keys)
    return jnp.mean(gb, axis=0), jnp.mean(gw, axis=0)

def to_visible(actions):
    B = len(actions)
    d = np.zeros((B, VIS_N), dtype=bool)
    d[np.arange(B), actions] = True
    return d

def train(h_dyn, acts, n_epochs=30, batch=64, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 0.1, N).astype(np.float32)
    w = rng.normal(0, 0.1, len(eidx)).astype(np.float32)
    # Freeze temporal at zero
    w[~learn_mask] = 0.0
    opt = optax.adam(lr)
    ostate = opt.init((b, w))
    n = len(acts)
    data_all = to_visible(acts)
    key = jax.random.key(seed + 999)
    freeze = jnp.asarray(~learn_mask)  # True where frozen
    for ep in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            bb = jnp.asarray(h_dyn[idx]); dd = jnp.asarray(data_all[idx])
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, len(idx))
            gb, gw = batch_grads(jnp.asarray(b), jnp.asarray(w), bb, dd, keys)
            # Zero gradients for frozen weights
            gw = jnp.where(freeze, 0.0, gw)
            upd, ostate = opt.update((gb, gw), ostate, (b, w))
            b, w = optax.apply_updates((b, w), upd)
            b = jnp.clip(b, -Z1_MAX, Z1_MAX)
            w = jnp.clip(w, -Z1_MAX, Z1_MAX)
            w = jnp.where(freeze, 0.0, w)  # enforce freeze
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{n_epochs}", flush=True)
    return np.asarray(b), np.asarray(w)

if __name__ == '__main__':
    d = np.load('s2_data.npz')
    t0 = time.time()
    b, w = train(d['train_h_dyn'], d['train_act'], n_epochs=30, seed=1)
    print(f"done in {time.time()-t0:.0f}s")
    np.savez('s2_theta_frozen.npz', biases=b, weights=w, edges=np.array(eidx))
    print("saved s2_theta_frozen.npz")
