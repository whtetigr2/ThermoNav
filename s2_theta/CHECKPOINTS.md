# s2_theta checkpoints — what each .npz is

**Ships:** `s2_theta_v2.npz` — the trained 123-param kernel (18 biases + 105
couplings). Default in `s2_learned.py`. Human-readable in `params.txt`.

## Training data

| File | What |
|---|---|
| `s2_data_v2_full.npz` | Full v2 training set |
| `s2_data_v2.npz` | v2 training subset |
| `s2_data_v2_partial.npz` | Partial (ablation) |
| `s2_data_v2_pilot.npz` | Pilot run |
| `s2_data_v2b.npz` | v2b variant |
| `s2_data.npz` / `s2_data_partial.npz` | Round-1 data |

## Model checkpoints (provenance, not for flight)

| File | What |
|---|---|
| `s2_theta.npz` | Round-1 model (superseded by v2) |
| `s2_theta_teacher.npz` | Engineered teacher params |
| `s2_theta_frozen.npz` / `train_frozen.py` | Frozen-ablation |
| `s2_theta_hybrid.npz` | Hybrid variant (ablation) |
| `s2_theta_random.npz` / `s2_theta_v2_random.npz` | Random inits |
| `s2_theta_warm.npz` / `s2_theta_v2_warm.npz` | Warm-start variants (warm-start hurts — see S2_THETA.md) |
| `s2_theta_v2_random_pilot.npz` / `s2_theta_v2_warm_pilot.npz` | Pilot checkpoints |
| `temporal_freeze_mask.npy` | Training artifact |

The v4 retrain (heavy-fog targeted) was killed — worse agreement, no outcome
gain. Not shipped, not stored here. See [`../REDTEAM.md`](../REDTEAM.md).
