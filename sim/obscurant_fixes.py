"""Obscurant fixes for Phase 4's silent D5-blindness (DYNAMICS_PHASE4.md §6).

Three fixes, LEGO order (each validated before the next). New files only;
Phases 1-4 untouched. All parameters documented with provenance.

BACKGROUND: at whiteout (V≈10 m) attenuation erases wall returns, walls read
unobserved (marg≈0.5, below D5's denial), D5 approves, drone flies into real
walls at steps 7-25. THRML graded: whiteout 2/8, 6 collisions. D5 refusals
4.5 (clear) → 0.5 (whiteout) — the failure is SILENT. Sampler never traps
(0/33); it confidently tracks a lie.

FIX 1 — Visibility-gated D5 (the silent killer): when the sensor is blind,
  unobserved voxels read as BLOCKED, not empty. Uses the global visibility
  metric (per-ray range consistency, shared with FIX 3): if vis < 0.50,
  D5 denies ALL unobserved voxels. When visibility is good, D5 is unchanged.

FIX 2 — Phantom confirmation gating: backscatter phantoms are transient
  (particles move); real obstacles persist. A hit only earns full l_occ
  after confirmation (hit_conf >= CONFIRM_K, i.e. seen in ~2 recent scans);
  unconfirmed hits get provisional l_occ * PROVISIONAL_SCALE. Must not slow
  legitimate wall latching beyond usefulness (walls confirm on scan 2).

FIX 3 — Low-vis executive: estimate visibility online from per-ray range
  consistency (fraction of returns matching the previous scan within 1 m).
  Measured [simulation]: clear 1.00 / light 0.96 / moderate 0.86 /
  heavy 0.61 / whiteout 0.39. Below VIS_ENTER (0.50) for 3 consecutive
  scans -> stop-and-hover (hold position, keep mapping); resume above
  VIS_EXIT (0.60). Skip rate is NOT used: it is non-monotonic
  (heavy 20.6% vs whiteout 20.3% — backscatter "rescues" rays from
  skipping at whiteout).

INTEGRITY: fixes touch only the dynamics layer's sensor integration and D5
wrapping. GT occupancy is never consulted. planner.py untouched.
"""
import numpy as np

import planner

# ---------------------------------------------------------------- FIX 1
# Visibility-gated D5: when the sensor is blind, unobserved voxels read as
# BLOCKED, not empty. Uses the global visibility metric (range-consistency,
# shared with FIX 3). Denies ALL unobserved voxels when vis < VIS_ENTER.
# (An earlier per-voxel attenuation-evidence design failed: voxels near the
# drone accumulate atten_ev from every skipped ray regardless of direction,
# and per-direction skip fractions are washed out by backscatter. The global
# visibility signal is the honest one.)
VIS_FIX1_ENTER = 0.50     # same threshold as FIX 3 [measured]

# ---------------------------------------------------------------- FIX 2
CONFIRM_DECAY = 0.6      # per-scan decay of hit confirmation [tuned]
CONFIRM_K = 1.5          # confirmed if hit_conf >= K (~2 recent scans)
                         # [tuned: real walls confirm on scan 2]
PROVISIONAL_SCALE = 0.3  # l_occ multiplier for unconfirmed hits [tuned]
VIS_CARVE_GATE = 0.70    # only gate carving when vis below this [tuned:
                         # clear-air vis ~0.7+ → normal carving; degraded
                         # → confirmation required. Prevents clear-air
                         # regression from 1-scan mapping delay.]

# ---------------------------------------------------------------- FIX 3
# Visibility = range consistency (temporal) AND far-hit fraction (spatial).
# Consistency alone false-triggers on heavy during motion (drone movement
# breaks range matching). Far-hit fraction is motion-robust: phantoms are
# near-field (<10 m), true walls return from afar. Measured [simulation,
# moving drone]: clear 0.19 / moderate 0.15 / heavy 0.04 / whiteout 0.00.
# Hover requires BOTH signals blind.
VIS_ENTER = 0.50         # consistency below this [measured: heavy
                         # 0.61 (static), whiteout 0.39 — loose threshold
                         # prioritizes whiteout safety; FPs bounded by
                         # temporary hover]
VIS_EXIT = 0.60          # resume above this (hysteresis) [design]
FAR_ENTER = 0.02         # far-hit fraction below this [measured: heavy
                         # 0.04, whiteout 0.00]
FAR_EXIT = 0.05          # resume above this (hysteresis) [design]
VIS_TRIGGER_SCANS = 3    # consecutive sub-threshold scans to trigger
                         # [design: fast response prioritized]
HOVER_MAX_SCANS = 5      # max consecutive hover scans before forced resume
                         # [design: bounds false-positive damage — a false
                         # trigger costs 5 s, not the mission]
VIS_SMOOTH_A = 0.7       # exponential smoothing factor [design choice]
CONSIST_TOL = 1.0        # m: range-match tolerance for consistency
                         # [assumed: well above 0.15 m range noise]
FAR_RANGE = 10.0         # m: "far" hit threshold [design: beyond phantom
                         # clip range]


# ---------------------------------------------------------------- FIX 4
# Stale-map stand-down: D5's vetoes are only as good as the map they see.
# When the published map is BOTH degraded (fog band, live tracker signal)
# AND stale (pipeline freeze), stand down (hover) instead of deciding on
# a lie. Conjunctive by measured necessity: a staleness-only gate would
# fire on clear-air died_8 (6/14, 0 collisions — fails safe already),
# pure false-positive cost, zero safety gain. Host-layer classical;
# no TSU/kernel/Z1 impact.
STALE_VIS_ENTER = 0.70   # degraded band: vis below this [measured:
                         # moderate 0.86 (out), heavy 0.61 (in)]
STALE_FAR_ENTER = 0.10   # and far-hit fraction below this [measured:
                         # moderate 0.15 (out), heavy 0.04 (in)]
STALE_VIS_EXIT = 0.75    # band exit (hysteresis) [design]
STALE_FAR_EXIT = 0.15    # band exit (hysteresis) [design]
STALE_AGE_S = 1.0        # map staleness trigger, sim-seconds [measured:
                         # collision freeze window was 2.0 s; 1.0 s is
                         # a 2x engineering margin under it]
STALE_SKIP_ENTER = 0.15  # degraded band: skip fraction above this
                         # [measured: clear 0.005, moderate 0.33, heavy
                         # 0.54, whiteout 0.30 — huge margin both ways.
                         # Skip rate is motion- and geometry-robust where
                         # vis/far are not: clear-air motion dips vis to
                         # 0.69 and tight rooms dip far to 0.045, which
                         # false-latched the vis/far-only band.]


class EvidenceTracker:
    """Per-scan evidence for the three fixes. One instance per mission."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.hit_conf = np.zeros(shape, dtype=float)   # FIX 2
        self.prev_ranges = {}                          # FIX 3: ray -> range
        self.vis_smooth = 1.0                          # FIX 1/3: consistency
        self.far_smooth = 1.0                          # FIX 3: far-hit frac
        self.lowvis_count = 0                          # FIX 3
        self.hover = False                             # FIX 3 output
        self.hover_scans = 0                           # FIX 3: current hover
        self.n_scans = 0
        self.n_hover_scans = 0                         # stats
        self.refusals_at_scan = []                     # FIX 3: D5 health
        self.stale_band = False      # FIX 4: latched degraded-perception band
        self.stale_hover = False     # FIX 4 output: stand down on stale map
        self.n_stale_hover_scans = 0  # stats
        self.skip_smooth = 0.0       # FIX 4: smoothed ray skip fraction
                                     # (fog signal; starts clear)

    def _in_bounds(self, v):
        return (0 <= v[0] < self.shape[0] and 0 <= v[1] < self.shape[1]
                and 0 <= v[2] < self.shape[2])

    def update(self, scans, fix1=True, fix2=True, fix3=True, n_refusals=0):
        """Process one scan's 5-tuples. Returns (n_returns, n_consistent).
        
        n_refusals: cumulative D5 move refusals (for FIX 3 health gate).
        """
        # Visibility (FIX 1/3) is always computed when any fix is on —
        # it's the shared blindness signal.
        # FIX 2: freeze hit_conf while hovering — a stationary drone gets
        # repeat phantom hits that would spuriously confirm. (Whiteout
        # hover collision: phantoms confirmed during hover, then carved.)
        was_hovering = self.hover
        if fix2 and not was_hovering:
            self.hit_conf *= CONFIRM_DECAY
        # FIX 3 D5-health: track refusal rate. Whiteout signature is D5
        # going QUIET (blind, not refusing); heavy has ACTIVE refusals.
        self.refusals_at_scan.append(n_refusals)
        if len(self.refusals_at_scan) > 6:
            self.refusals_at_scan.pop(0)
        n_ret, n_con = 0, 0
        for i, t in enumerate(scans):
            origin, direction, meas_r, sensor_hit, has_return = t[:5]
            if not has_return:
                self.prev_ranges.pop(i, None)
                continue
            n_ret += 1
            consistent = (i in self.prev_ranges
                          and abs(meas_r - self.prev_ranges[i]) < CONSIST_TOL)
            self.prev_ranges[i] = meas_r
            if consistent:
                n_con += 1
            if fix2 and sensor_hit and not was_hovering:
                p = np.asarray(origin, dtype=float) \
                    + np.asarray(direction, dtype=float) * meas_r
                v = (int(p[0]), int(p[1]), int(p[2]))
                if self._in_bounds(v):
                    self.hit_conf[v] += 1.0
        # FIX 1/3 shared visibility metrics (always updated when tracking).
        if n_ret > 0:
            vis = n_con / n_ret
            self.vis_smooth = (VIS_SMOOTH_A * self.vis_smooth
                               + (1 - VIS_SMOOTH_A) * vis)
        # Far-hit fraction (motion-robust blindness signal).
        n_hit, n_far = 0, 0
        for t in scans:
            if t[3]:  # sensor_hit
                n_hit += 1
                if t[2] > FAR_RANGE:
                    n_far += 1
        if n_hit > 0:
            far = n_far / n_hit
            self.far_smooth = (VIS_SMOOTH_A * self.far_smooth
                               + (1 - VIS_SMOOTH_A) * far)
        # FIX 4: ray skip fraction (fog signal). Motion- and geometry-
        # robust: attenuation kills returns regardless of drone motion or
        # room size, where vis dips under motion and far dips in tight
        # rooms. Measured [simulation, static]: clear 0.005, moderate
        # 0.33, heavy 0.54, whiteout 0.30.
        n_scan = len(scans)
        if n_scan > 0:
            n_skip = sum(1 for t in scans if not t[4])
            skip = n_skip / n_scan
            self.skip_smooth = (VIS_SMOOTH_A * self.skip_smooth
                                + (1 - VIS_SMOOTH_A) * skip)
        if fix3:
            blind = (self.vis_smooth < VIS_ENTER
                     and self.far_smooth < FAR_ENTER)
            clear = (self.vis_smooth > VIS_EXIT
                     and self.far_smooth > FAR_EXIT)
            # D5-health gate: only hover if D5 has gone quiet. At heavy,
            # D5 refuses actively (healthy caution); at whiteout, D5 goes
            # silent (blind). Require <2 refusals in last 5 scans.
            d5_quiet = True
            if len(self.refusals_at_scan) >= 6:
                recent = (self.refusals_at_scan[-1]
                          - self.refusals_at_scan[-6])
                d5_quiet = recent < 2
            if blind and d5_quiet:
                self.lowvis_count += 1
            elif clear:
                self.lowvis_count = 0
            # hysteresis: enter when both blind, exit when both clear.
            # Hover is TEMPORARY (HOVER_MAX_SCANS): bounds false-positive
            # damage. If still blind after resume, it re-triggers.
            if not self.hover and self.lowvis_count >= VIS_TRIGGER_SCANS:
                self.hover = True
                self.hover_scans = 0
            elif self.hover:
                self.hover_scans += 1
                self.n_hover_scans += 1
                if clear or self.hover_scans >= HOVER_MAX_SCANS:
                    self.hover = False
                    self.lowvis_count = 0
                    self.hover_scans = 0
        self.n_scans += 1
        return n_ret, n_con

    def update_stale(self, map_age_s, fix4=True):
        """FIX 4: conjunctive stale-map stand-down. Call once per outer
        step, AFTER update() (needs vis_smooth/far_smooth).

        map_age_s: sim-seconds since the last fresh publish (0.0 when no
        jitter is configured — the gate then never fires, correctly).
        Sets self.stale_hover: True while the map is BOTH stale
        (map_age_s >= STALE_AGE_S) AND perception is in the degraded band.
        Direct signal-following (no scan counting): hover chains through
        sustained freezes, releases on the first fresh publish.
        """
        if not fix4:
            self.stale_hover = False
            return
        # Degraded band with hysteresis (live tracker signals, shared
        # with FIX 1/3 — the sensor keeps scanning during a freeze, so
        # these see reality even when the published map is stale).
        # Triple conjunct: vis+far alone false-latch in clear tight rooms
        # under motion (vis dips to 0.69, far to 0.045 [measured]); skip
        # rate is the motion/geometry-robust fog discriminator.
        degraded = (self.vis_smooth < STALE_VIS_ENTER
                    and self.far_smooth < STALE_FAR_ENTER
                    and self.skip_smooth > STALE_SKIP_ENTER)
        clear = (self.vis_smooth > STALE_VIS_EXIT
                 and self.far_smooth > STALE_FAR_EXIT)
        if degraded:
            self.stale_band = True
        elif clear:
            self.stale_band = False
        self.stale_hover = bool(self.stale_band
                                and map_age_s >= STALE_AGE_S)
        if self.stale_hover:
            self.n_stale_hover_scans += 1


def integrate_scan_fixes(logodds, occ_shape, scans, tracker,
                         l_occ=0.60, l_free=-0.73,
                         l_min=-4.0, l_max=4.0, step=0.5, hit_radius=0.75,
                         max_range=40.0, obs_count=None, fix2=True):
    """Inverse sensor model with FIX 2 (confirmation-gated evidence).

    A hit only earns full evidence if tracker.hit_conf[v] >= CONFIRM_K
    (confirmed across scans — persistent surface, not a transient particle).
    Unconfirmed hits get provisional l_occ and, critically, NO free-space
    carving: a phantom hit beyond an attenuated wall must not carve the
    wall voxel as free (Phase-4 whiteout collisions: phantom at 5 m carved
    through the real wall at 2 m whose return was attenuated).
    No-return rays are skipped (no information), as in Phase 4.

    tracker.update(scans) MUST be called before this (it increments
    hit_conf).
    """
    nx, ny, nz = occ_shape
    n_hits, n_skipped = 0, 0
    for tup in scans:
        if len(tup) == 5:
            origin, direction, meas_r, sensor_hit, has_return = tup
        else:  # 4-tuple (production format): every ray has a return
            origin, direction, meas_r, true_r = tup
            sensor_hit, has_return = bool(true_r < max_range - 1e-6), True
        if not has_return:
            n_skipped += 1
            continue
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        # FIX 2: check confirmation BEFORE carving. Unconfirmed (phantom)
        # hits do not carve free space — but only when visibility is
        # degraded (VIS_CARVE_GATE). In clear air, hits are real and
        # carving is safe; gating there causes a 1-scan mapping delay
        # regression.
        confirmed = True
        if fix2 and sensor_hit and tracker.vis_smooth < VIS_CARVE_GATE:
            p_hit = o + d * meas_r
            v_hit = (int(p_hit[0]), int(p_hit[1]), int(p_hit[2]))
            if (0 <= v_hit[0] < nx and 0 <= v_hit[1] < ny
                    and 0 <= v_hit[2] < nz):
                confirmed = tracker.hit_conf[v_hit] >= CONFIRM_K
            else:
                confirmed = False
        if confirmed:
            t = step
            while t < meas_r - hit_radius:
                p = o + d * t
                ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    logodds[ix, iy, iz] = np.clip(
                        logodds[ix, iy, iz] + l_free, l_min, l_max)
                    if obs_count is not None:
                        obs_count[ix, iy, iz] += 1
                else:
                    break
                t += step
        if sensor_hit:
            n_hits += 1
            p = o + d * meas_r
            v = (int(p[0]), int(p[1]), int(p[2]))
            if 0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz:
                if (fix2 and not confirmed
                        and tracker.vis_smooth < VIS_CARVE_GATE):
                    l_use = l_occ * PROVISIONAL_SCALE
                else:
                    l_use = l_occ
                logodds[v] = np.clip(logodds[v] + l_use, l_min, l_max)
                if obs_count is not None:
                    obs_count[v] += 1
    return logodds, n_hits, n_skipped


def safe_to_enter_fixes(voxel, marg, obs_mask, pos_v, tracker, fix1=True):
    """D5 wrapper with FIX 1 (visibility-gated denial).

    When the sensor is blind (vis_smooth < VIS_FIX1_ENTER), unobserved
    voxels read as BLOCKED, not empty: deny them all. When visibility is
    good, fall through to planner.safe_to_enter unchanged.
    planner.py untouched.
    """
    v = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
    nx, ny, nz = obs_mask.shape
    if not (0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz):
        # Out-of-bounds S2 proposal: deny, matching planner.safe_to_enter's
        # documented contract ("Returns False for: out-of-bounds voxels").
        # Without this, the FIX 1 branch below indexed obs_mask[v] on an
        # off-grid proposal and raised IndexError — seen in whiteout,
        # where vis_smooth < VIS_FIX1_ENTER activates this branch. In clear
        # air the branch never executes, so behavior there is unchanged.
        return False
    if fix1 and tracker.vis_smooth < VIS_FIX1_ENTER:
        if not bool(obs_mask[v]):
            return False
    return planner.safe_to_enter(voxel, marg, obs_mask, pos_v)
