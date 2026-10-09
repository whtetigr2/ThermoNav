# thermopilot/ — navigation harness

The classical host side of the stack: the mission executive that the
thermodynamic kernel plugs into.

- `harness.py` — `S2Agent` (engineered teacher: 6 directions × 3-step lookahead),
  A* global planner, D5 safety veto (5-gate: veto, peek/refusal/escape,
  blind-spot denial, freshness gate)
- `ising.py` — Ising model helpers (`one_hot`, `linear_costs`, `bilinear`,
  `z1_check`)
- `gibbs_ising.py` — the Gibbs sampler used by the S2 decision kernel

The trained kernel lives in `../s2_theta/` (`s2_theta_v2.npz`, 123 params,
drop-in via `s2_learned.py`). Claim: Z1-legal formulation, validated in
simulation — never silicon, never real aircraft.
