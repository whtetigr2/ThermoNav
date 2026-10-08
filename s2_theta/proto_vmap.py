"""Prototype: vmap'd estimate_kl_grad with per-sample biases."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp, numpy as np
from thrml import Block, SpinNode, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init
import sys; sys.path.insert(0, '.')
from model_def import build_nodes_edges, N, VIS_N

nodes, edges, eidx, vis_nodes, free_blocks, clamped_blocks, h_eng, J_eng = build_nodes_edges()
beta = jnp.array(1.0)
sched = SamplingSchedule(5, 10, 2)

# engineered weights on mask
w_eng = np.array([J_eng[i, j] for (i, j) in eidx], dtype=np.float32)
b_eng = h_eng.astype(np.float32)

def single_grad(biases, weights, h_dyn, data, key):
    """One sample's KL grad. biases: (18,) learnable; h_dyn: (18,) fixed."""
    model = IsingEBM(nodes, edges, biases + h_dyn, weights, beta)
    spec = IsingTrainingSpec(model, [Block(vis_nodes)], [],
                             clamped_blocks, free_blocks, sched, sched)
    k1, k2, k3 = jax.random.split(key, 3)
    init_pos = hinton_init(k1, model, clamped_blocks, (1, 1))
    init_neg = hinton_init(k2, model, free_blocks, (1,))
    gw, gb, _, _ = estimate_kl_grad(k3, spec, nodes, edges, [data[None, :]], [],
                                    init_pos, init_neg)
    return gb, gw

# fake batch: 4 samples
rng = np.random.default_rng(0)
B = 4
h_dyn_b = rng.normal(0, 0.25, (B, N)).astype(np.float32)
data_b = np.zeros((B, VIS_N), dtype=bool)
for b in range(B):
    data_b[b, rng.integers(0, VIS_N)] = True

batched = jax.jit(jax.vmap(single_grad, in_axes=(None, None, 0, 0, 0)))
keys = jax.random.split(jax.random.key(0), B)
gb, gw = batched(jnp.asarray(b_eng), jnp.asarray(w_eng),
                 jnp.asarray(h_dyn_b), jnp.asarray(data_b), keys)
print('grad_b shape:', gb.shape, 'grad_w shape:', gw.shape)
print('mean |grad_b|:', float(jnp.abs(gb).mean()), 'mean |grad_w|:', float(jnp.abs(gw).mean()))
print('PROTOTYPE OK')
