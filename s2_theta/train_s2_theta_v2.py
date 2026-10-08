"""Round 2: n-step realized-future supervision. Fully-visible conditional Ising.

All 18 spins clamped in the positive phase to the expert's realized
(a_t, a_{t+1}, a_{t+2}) one-hots. Positive moments are EXACT from data
(no positive-phase sampling). Negative phase: THRML block-Gibbs on all 18
spins with per-sample fields h_total = h_static(theta) + h_dyn(x), vmapped.

Manual KL gradient (fully-visible BM, exact):
    dKL/db_i    = <s_i>_model    - <s_i>_data
    dKL/dw_ij   = <s_i s_j>_model - <s_i s_j>_data   (on Z1 mask)
passed to optax (descent), same convention as round 1's estimate_kl_grad.

Z1 projection after each step: |h|,|J| <= 6.0 (mask enforces degree <= 16).
New files only; round-1 artifacts untouched.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from thrml import SamplingSchedule
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml import sample_states
sys.path.insert(0, '.')
from model_def import build_nodes_edges, N, K, T

nodes, edges, eidx, vis_nodes, free_blocks, _cb, h_eng, J_eng = build_nodes_edges()
beta = jnp.array(1.0)
SCHED = SamplingSchedule(16, 32, 2)  # warmup, samples, steps_per_sample
Z1_MAX = 6.0

w_eng = np.array([J_eng[i, j] for (i, j) in eidx], dtype=np.float32)
b_eng = h_eng.astype(np.float32)

# static index maps (closed over by jitted fns)
_node_index = {n: i for i, n in enumerate(nodes)}
_block_idx = [jnp.array([_node_index[n] for n in blk.nodes]) for blk in free_blocks]
_ei = jnp.array([i for (i, j) in eidx])
_ej = jnp.array([j for (i, j) in eidx])
N_E = len(eidx)


def seq_to_bool(seq):
    """seq (M,3) int actions -> (M,18) bool (three one-hots, True=+1)."""
    M = len(seq)
    d = np.zeros((M, N), dtype=bool)
    for t in range(T):
        d[np.arange(M), t * K + seq[:, t]] = True
    return d


def neg_moments(biases, weights, h_dyn, key):
    """Model moments via THRML block Gibbs. Returns (neg_b (18,), neg_w (105,))."""
    model = IsingEBM(nodes, edges, biases + h_dyn, weights, beta)
    prog = IsingSamplingProgram(model, free_blocks, [])
    k1, k2 = jax.random.split(key)
    init = hinton_init(k1, model, free_blocks, ())
    states = sample_states(k2, prog, SCHED, init, [], free_blocks)
    full = jnp.zeros((SCHED.n_samples, N))
    for blk_states, idx in zip(states, _block_idx):
        full = full.at[:, idx].set(blk_states.astype(jnp.float32))
    s = 2.0 * full - 1.0
    nb = s.mean(axis=0)
    nw = (s[:, _ei] * s[:, _ej]).mean(axis=0)
    return nb, nw


@jax.jit
def batch_neg_moments(biases, weights, h_dyn_b, keys):
    nb, nw = jax.vmap(neg_moments, in_axes=(None, None, 0, 0))(
        biases, weights, h_dyn_b, keys)
    return nb.mean(axis=0), nw.mean(axis=0)


def train(init_name, h_dyn, seq, n_epochs=30, batch=64, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    if init_name == 'warm':
        b = b_eng.copy(); w = w_eng.copy()
    elif init_name == 'random':
        b = rng.normal(0, 0.1, N).astype(np.float32)
        w = rng.normal(0, 0.1, N_E).astype(np.float32)
    else:
        raise ValueError(init_name)
    data_b = seq_to_bool(seq)  # (M,18) bool
    opt = optax.adam(lr)
    ostate = opt.init((b, w))
    n = len(seq)
    key = jax.random.key(seed + 4242)
    losses = []
    s_pm_all = 2.0 * data_b.astype(np.float32) - 1.0
    for ep in range(n_epochs):
        perm = rng.permutation(n)
        egb, egw, nb = 0.0, 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            bb = jnp.asarray(h_dyn[idx])
            sb = jnp.asarray(s_pm_all[idx])
            # exact positive moments
            pos_b = sb.mean(axis=0)
            pos_w = (sb[:, _ei] * sb[:, _ej]).mean(axis=0)
            # THRML negative moments
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, len(idx))
            neg_b, neg_w = batch_neg_moments(jnp.asarray(b), jnp.asarray(w), bb, keys)
            # KL grads (descent convention, matches round 1)
            gb = neg_b - pos_b
            gw = neg_w - pos_w
            upd, ostate = opt.update((gb, gw), ostate, (b, w))
            b, w = optax.apply_updates((b, w), upd)
            b = jnp.clip(b, -Z1_MAX, Z1_MAX)
            w = jnp.clip(w, -Z1_MAX, Z1_MAX)
            egb += float(jnp.abs(gb).mean()); egw += float(jnp.abs(gw).mean()); nb += 1
        # epoch diagnostic: mean |temporal J|
        losses.append((egb / nb, egw / nb))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{init_name}] epoch {ep+1}/{n_epochs} "
                  f"mean|gb|={egb/nb:.4f} mean|gw|={egw/nb:.4f}", flush=True)
    return np.asarray(b), np.asarray(w), np.array(losses)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = np.load('s2_data_v2.npz')
    print(f"data: train {len(d['train_seq'])} windows, held {len(d['held_seq'])}", flush=True)
    for init in ['warm', 'random']:
        t0 = time.time()
        b, w, losses = train(init, d['train_h_dyn'], d['train_seq'],
                             n_epochs=args.epochs, batch=args.batch,
                             lr=args.lr, seed=args.seed)
        print(f"[{init}] done in {time.time()-t0:.0f}s", flush=True)
        np.savez(f's2_theta_v2_{init}.npz', biases=b, weights=w,
                 edges=np.array(eidx), h_eng=b_eng, w_eng=w_eng,
                 losses=np.array(losses))
    print("saved s2_theta_v2_warm.npz, s2_theta_v2_random.npz")


if __name__ == '__main__':
    main()
