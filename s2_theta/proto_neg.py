import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys; sys.path.insert(0, '.')
import jax, jax.numpy as jnp, numpy as np
from thrml import SamplingSchedule
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml import sample_states
from model_def import build_nodes_edges, N

nodes, edges, eidx, vis_nodes, free_blocks, clamped_blocks, h_eng, J_eng = build_nodes_edges()
beta = jnp.array(1.0)
sched = SamplingSchedule(8, 16, 2)
w_eng = jnp.asarray([J_eng[i,j] for (i,j) in eidx], dtype=jnp.float32)
b_eng = jnp.asarray(h_eng, dtype=jnp.float32)
h_dyn = jnp.zeros(N)

model = IsingEBM(nodes, edges, b_eng + h_dyn, w_eng, beta)
prog = IsingSamplingProgram(model, free_blocks, [])
key = jax.random.key(0)
k1, k2 = jax.random.split(key)
init = hinton_init(k1, model, free_blocks, ())
print('n free_blocks:', len(free_blocks), 'init shapes:', [a.shape for a in init])
states = sample_states(k2, prog, sched, init, [], free_blocks)
print('n returned:', len(states), 'shapes:', [a.shape for a in states])
print('dtype:', states[0].dtype)
