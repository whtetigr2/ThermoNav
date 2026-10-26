"""Train S2-theta: learn (h_static, J) from expert demonstrations via THRML.

Design (frozen):
- Visible: t=0's 6 spins, clamped to expert's one-hot action.
- Latent: t=1,2's 12 spins, sampled in positive phase.
- Conditioning: h_dyn(x) added to learned biases per-sample (vmap over batch).
- Params: 18 biases + 105 weights (Z1 mask) = 123. Clip |.|<=6 after each step.
- Inits: warm-start (engineered) and small-random. Train both, keep better.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
import optax
from thrml import Block, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init
sys.path.insert(0, '.')
from model_def import build_nodes_edges, N, VIS_N, K

nodes, edges, eidx, vis_nodes, free_blocks, clamped_blocks, h_eng, J_eng = build_nodes_edges()
beta = jnp.array(1.0)
SCHED = SamplingSchedule(8, 16, 2)  # warmup, samples, steps_per_sample
Z1_MAX = 6.0

w_eng = np.array([J_eng[i, j] for (i, j) in eidx], dtype=np.float32)
b_eng = h_eng.astype(np.float32)


def single_grad(biases, weights, h_dyn, data, key):
    model = IsingEBM(nodes, edges, biases + h_dyn, weights, beta)
    spec = IsingTrainingSpec(model, [Block(vis_nodes)], [],
                             clamped_blocks, free_blocks, SCHED, SCHED)
    k1, k2, k3 = jax.random.split(key, 3)
    init_pos = hinton_init(k1, model, clamped_blocks, (1, 1))
    init_neg = hinton_init(k2, model, free_blocks, (1,))
    gw, gb, _, _ = estimate_kl_grad(k3, spec, nodes, edges, [data[None, :]], [],
                                    init_pos, init_neg)
    return gb, gw


@jax.jit
def batch_grads(biases, weights, h_dyn_b, data_b, keys):
    gb, gw = jax.vmap(single_grad, in_axes=(None, None, 0, 0, 0))(
        biases, weights, h_dyn_b, data_b, keys)
    return jnp.mean(gb, axis=0), jnp.mean(gw, axis=0)


def to_visible(actions):
    """actions (B,) int -> (B, 6) bool one-hot (True=+1)."""
    B = len(actions)
    d = np.zeros((B, VIS_N), dtype=bool)
    d[np.arange(B), actions] = True
    return d


def train(init_name, h_dyn, acts, n_epochs=30, batch=64, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    if init_name == 'warm':
        b = b_eng.copy(); w = w_eng.copy()
    elif init_name == 'random':
        b = (rng.normal(0, 0.1, N)).astype(np.float32)
        w = (rng.normal(0, 0.1, len(eidx))).astype(np.float32)
    else:
        raise ValueError(init_name)
    opt = optax.adam(lr)
    ostate = opt.init((b, w))
    n = len(acts)
    data_all = to_visible(acts)
    key = jax.random.key(seed + 999)
    losses = []
    for ep in range(n_epochs):
        perm = rng.permutation(n)
        ep_gb, ep_gw, nb = 0.0, 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            bb = jnp.asarray(h_dyn[idx]); dd = jnp.asarray(data_all[idx])
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, len(idx))
            gb, gw = batch_grads(jnp.asarray(b), jnp.asarray(w), bb, dd, keys)
            upd, ostate = opt.update((gb, gw), ostate, (b, w))
            b, w = optax.apply_updates((b, w), upd)
            # Z1 projection
            b = jnp.clip(b, -Z1_MAX, Z1_MAX)
            w = jnp.clip(w, -Z1_MAX, Z1_MAX)
            ep_gb += float(jnp.abs(gb).mean()); ep_gw += float(jnp.abs(gw).mean()); nb += 1
        losses.append((ep_gb / nb, ep_gw / nb))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{init_name}] epoch {ep+1}/{n_epochs} "
                  f"mean|gb|={ep_gb/nb:.4f} mean|gw|={ep_gw/nb:.4f}", flush=True)
    return np.asarray(b), np.asarray(w), losses


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = np.load('s2_data.npz')
    out = {}
    for init in ['warm', 'random']:
        t0 = time.time()
        b, w, losses = train(init, d['train_h_dyn'], d['train_act'],
                             n_epochs=args.epochs, batch=args.batch,
                             lr=args.lr, seed=args.seed)
        print(f"[{init}] done in {time.time()-t0:.0f}s", flush=True)
        out[init] = (b, w)
        # The random-init checkpoint is saved as s2_theta_random_init.npz
        # (renamed 2026-10-26: "random" names the initialization of a
        # trained model, not an untrained control).
        fname = ('s2_theta_random_init.npz' if init == 'random'
                 else f's2_theta_{init}.npz')
        np.savez(fname, biases=b, weights=w,
                 edges=np.array(eidx), h_eng=b_eng, w_eng=w_eng,
                 losses=np.array(losses))
    print("saved s2_theta_warm.npz, s2_theta_random_init.npz")


if __name__ == '__main__':
    main()
