"""Dynamic-object layer v2: previous-scan differencing + CV tracks.

Phase 2(a) of the roadmap burn-down (2026-10-11). Successor to the
motion LEGO (motion.py, 30/48) and the dodge LEGO (dodge.py, 26/48).
Those results diagnosed the binding constraints, in their own words:

1. Detection was ROOM-BRITTLE: motion.py fires only in voxels the
   ACCUMULATED static map believes are deeply free (log-odds < -1.0).
   Carving depth varies with room geometry — in indoorcd_128 the mover
   barely registered (2-6 events/mission) and the dodge never fired.
2. Trigger LATENCY: event (1 scan) + trail confirmation (1 scan) +
   velocity from the confirmed-trail least-squares fit (1-2 s more)
   = 2-4 s before any response. Fast/early encounters end first.
3. Veto-only geometry: the motion layer blocked only the trail's PAST
   voxels; the dodge patched the current-position case reactively.

This layer attacks (1) and (2) at the detection stage, and folds (3)'s
projection INTO the block query, keeping ONE mechanism:

DETECTION — appearance in either of two reference frames.
(a) PREVIOUS-SCAN channel: a hit is an appearance if the previous
    scan's rays passed through that voxel and hit something farther
    away (the previous scan carved it free — same 0.5 m sampling /
    0.75 m hit radius as the inverse sensor model). Single-scan
    carving is uniform wherever the sensor actually looked, so this
    channel does not depend on the room's accumulated carving depth
    (the 128 failure). Appearances adjacent (26-neighborhood) to a
    previous-scan STATIC hit are suppressed: a range-noise hit
    landing one voxel short of a wall sits next to that wall's
    previous hit; a mover in open space does not. Previous hits that
    were themselves appearances do NOT suppress — otherwise a slow
    mover's next step (adjacent to its own last position) would be
    filtered out by its own trail. (Caught by test_dynlayer.py.)
(b) ACCUMULATED-MAP channel (motion.py's rule, verbatim thresholds):
    a hit in a voxel the static map believes deeply free (log-odds
    < -1.0) with no occupied neighbor. This covers the one geometry
    channel (a) is blind to: a mover walking AWAY from the sensor
    steps into voxels its own body occluded in the previous scan —
    never carved by it — but that earlier scans carved before the
    mover arrived (and the map's carving asymmetry keeps carved).
    The union is clustered and tracked as one stream; per-channel
    counts are kept for diagnosis.

TRACKING — constant-velocity association, velocity at confirmation.
Appearances are clustered (26-connectivity) and associated to tracks
by nearest predicted position (gate 3.0 m: a 2.0 m/s mover moves 2 m
between 1 Hz scans, plus voxel quantization). A track is CONFIRMED
(veto-level) on its second associated appearance — and the same two
points already give a velocity (refined by least squares as more
points land). The old chain needed confirmation PLUS a separate
trail history before the dodge could estimate velocity; here
confirmation and velocity arrive together, ~1 scan after first
sighting. (v3: authority above veto level is tiered — see the fix
round notes below.)

BLOCKING — projected tube, not trail.
blocks(voxel) is True if the voxel lies within BLOCK_RADIUS of any
confirmed track's position projected forward over [0, PROJ_HORIZON]
under its estimated velocity. D5 therefore vetoes moves into where
the mover WILL BE, including the crossing case where the mover walks
into the drone's planned path — the gap the reactive dodge was built
to patch. The existing DodgeController still works unchanged: this
class exposes the same events/confirmed interface it consumes, fed
from confirmed tracks (latest observed voxel per scan).

Why a separate layer at all (condensation rule): the static occupancy
map CANNOT hold this state. Its inverse sensor model is tuned to
erase exactly this evidence (one +0.60 hit vs many-ray -0.73 carving —
validate_B_evidence.py measured the mover's voxel pinned at -3.40),
and tracks need object state (identity, velocity, age) that a
per-voxel log-odds field has no place to store. Merging into the map
would either contaminate static-obstacle performance (the map would
start believing transient hits) or reintroduce the erasure. The layer
therefore lives beside the map, reads only raw scans, and exports
only a block query + the trail interface. It never writes the map.

No ground-truth leakage: inputs are scans, the map's log-odds (for
channel (b) only), and sim time. Mover truth is never consulted.

Host-side classical component; no kernel/TSU involvement, no Z1
surface. Claim label: Z1-legal formulation, validated in simulation.

FIX ROUND v3 (2026-10-11, after red-team PASS WITH CAVEATS — see
docs/DYNLayer_REDTEAM.md on the moversrt branch):

F1 WALL-ADJACENCY: adjacency suppression no longer deletes a hit
    outright. A hit suppressed ONLY by wall adjacency becomes a
    wall-candidate, and candidates are chained across scans
    (track-before-detect): promotion to an appearance requires a
    3-scan chain of motion-consistent candidate clusters (steps
    0.4-2.2 m, direction cosine >= 0.7 between steps, 3rd point
    within 0.75 m of the constant-velocity prediction), and the
    promoted track is seeded with the chain points (a real
    velocity, not a fabrication). A 2-scan movement test was tried
    first and FAILED in closed loop (measured: 145 spurious
    promotions in one 004 crossing mission — the sparse rotating
    ray set makes static wall voxels flicker, and flicker centroids
    drift): single-step motion is not evidence in this sensor.
    Candidate voxels that were hits 2 scans ago are also dropped
    (flicker returns to the same voxel; a mover's leading edge was
    not occupied 2 scans ago). Once a chain promotes, it stays
    "hot": further consistent candidates promote immediately while
    the wall-hugger persists. A stationary candidate never promotes.
    Chain promotion grants veto authority only; full (dodge)
    authority needs a 4th consistent appearance (see _associate).
    Residual, measured and accepted: flicker chains still promote
    occasionally in candidate-dense rooms (071 no-mover baseline:
    41 promoted voxels, 94 motion refusals, mission success) — the
    wide-seed FP set bounds the rate; four other candidate
    discriminators were measured and rejected (next-scan fate,
    suppressor shadowing, cluster compactness, t-3 hit history —
    none separates flicker from mover in this sensor).
F2 TIERED AUTHORITY: confirmation is no longer one step.
    - veto authority (blocks() projected tube): 2 appearances AND
      fitted speed <= SPEED_ENVELOPE (2.6 m/s — staged movers are
      capped at 2.0; a 2-point fit above the envelope is treated as
      association noise, velocity untrusted, no tube at all).
    - full authority (the dodge-facing `confirmed` set): a 3rd
      appearance whose residual to the track's CV prediction is
      <= FULL_GATE_M, on top of the speed envelope. Two unrelated
      noise hits can still buy a local veto for one TTL window
      (bounded tax, measured), but they cannot trigger a dodge and
      cannot project a fabricated-velocity tube.
F3 MERGE / SELF-LOCKOUT: (i) a cluster contended by >= 2 tracks
    whose predictions are separated is SPLIT by nearest prediction
    before association, so crossing movers no longer donate one
    merged centroid to a single track; (ii) `prev_appeared` DECAYS
    over APPEAR_DECAY_S instead of being replaced per scan — one
    zero-appearance scan no longer turns a mover's own trail into
    its suppressor.
F4 TUBE OFFSET: blocks() compares the fitted position against
    voxel centers (v + 0.5, radius + 0.5) instead of truncating it
    with int() — a fit value of 9.9999999 for a mover at 10.0 used
    to shift the whole tube 1 voxel toward -x/-y, asymmetrically.
"""

import numpy as np

FREE_STEP = 0.5        # m: ray sampling for previous-scan free set —
                       # mirrors obscurant.integrate_scan exactly.
HIT_RADIUS = 0.75      # m: carving stops this far short of the hit —
                       # mirrors integrate_scan's hit_radius.
GATE_M = 3.0           # m: association gate around a track's predicted
                       # position. 2.0 m/s max mover x 1 s scan interval
                       # + ~1 m voxel/centroid slop.
CONFIRM_HITS = 2       # appearances before a track is actionable.
TRACK_TTL_S = 3.0      # s: drop a track unseen this long (a mover that
                       # left the sensor's view must not blockade forever).
FIT_WINDOW_S = 4.0     # s: position history used for the velocity fit.
SPEED_HI = 5.0         # m/s sanity cap (matches dodge.py); above this
                       # the estimate is treated as noise, velocity = 0.
FREE_THRESH = -1.0     # channel (b): log-odds below this = map believes
                       # deeply free (motion.py FREE_THRESH, verbatim).
OCC_LO = 1.0           # channel (b): neighbor log-odds above this =
                       # adjacent to a real obstacle -> suppress
                       # (motion.py OCC_LO, verbatim).
BLOCK_RADIUS = 1       # Chebyshev voxels around the projected tube.
PROJ_HORIZON_S = 2.0   # s: how far forward blocks() projects.
PROJ_STEP_S = 0.5      # s: projection sampling.
# --- fix-round v3 constants ---
SPEED_ENVELOPE = 2.6   # m/s: max fitted speed for ANY track authority.
                       # Staged movers cap at 2.0 m/s; the margin covers
                       # 2-point centroid slop. Above this the fit is
                       # association noise (red-team U4: +2.0/+2.0 m/s
                       # fabricated from two unrelated noise hits).
FULL_HITS = 3          # appearances before full (dodge) authority.
FULL_GATE_M = 1.5      # m: the appearance that grants full authority
                       # must land within this residual of the track's
                       # CV prediction (motion consistency, not just
                       # another hit inside the loose 3.0 m gate).
WALL_MIN_MOVE_M = 0.4  # m: min candidate-cluster step per scan (F1).
WALL_MAX_STEP_M = 2.2  # m: max candidate step per scan (2.0 m/s mover
                       # envelope + centroid slop).
WALL_CHAIN_RESID_M = 0.75  # m: 3rd chain point must land this close
                       # to the constant-velocity prediction.
WALL_CHAIN_COS = 0.7   # min direction cosine between successive
                       # candidate-chain steps.
APPEAR_DECAY_S = 2.5   # s: prev_appeared decay (F3) — a mover's own
                       # trail keeps not-suppressing it through a
                       # zero-appearance scan gap, within track TTL.


def _scan_geometry(scans, shape):
    """Split one scan into (hit voxel -> hit point, free voxel set),
    using the inverse sensor model's own carving convention."""
    nx, ny, nz = shape
    hits = {}
    free = set()
    for tup in scans:
        if len(tup) == 5:
            origin, direction, meas_r, sensor_hit, has_return = tup
        else:  # 4-tuple production format: every ray has a return
            origin, direction, meas_r, _true_r = tup
            sensor_hit, has_return = True, True
        if not has_return:
            continue
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        mr = float(meas_r)
        t = FREE_STEP
        while t < mr - HIT_RADIUS:
            p = o + d * t
            v = (int(p[0]), int(p[1]), int(p[2]))
            if 0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz:
                free.add(v)
            else:
                break
            t += FREE_STEP
        if sensor_hit:
            p = o + d * mr
            v = (int(p[0]), int(p[1]), int(p[2]))
            if 0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz:
                hits[v] = p
    free -= set(hits)
    return hits, free


def _clusters(voxels):
    """26-connected components of a voxel set."""
    rest = set(voxels)
    out = []
    while rest:
        seed = rest.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            a = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        w = (a[0] + dx, a[1] + dy, a[2] + dz)
                        if w in rest:
                            rest.discard(w)
                            comp.append(w)
                            stack.append(w)
        out.append(comp)
    return out


class _Track:
    __slots__ = ("points", "voxels", "n_hits", "t_last", "full_ok",
                 "_full_counted")

    def __init__(self, t, centroid, voxels):
        self.points = [(t, centroid)]  # (t, xyz) sub-voxel centroids
        self.voxels = list(voxels)     # latest observed voxels
        self.n_hits = 1
        self.t_last = t
        self.full_ok = False   # a >=3rd appearance landed motion-
                               # consistently (residual <= FULL_GATE_M)
        self._full_counted = False

    @property
    def confirmed(self):
        """Veto-level confirmation (2 appearances). NOTE (v3): this
        no longer implies dodge authority — see full_authority."""
        return self.n_hits >= CONFIRM_HITS

    def fit_speed(self):
        """Uncapped least-squares speed over the fit window, or None
        with < 2 points. (estimate() zeroes > SPEED_HI for display;
        authority decisions need the raw number.)"""
        pts = self.points
        if len(pts) < 2:
            return None
        ts = np.array([p[0] for p in pts])
        ps = np.array([p[1] for p in pts])
        if ts[-1] - ts[0] < 1e-9:
            return 0.0
        A = np.vstack([np.ones_like(ts), ts]).T
        vel = np.zeros(3)
        for i in range(3):
            coef, *_ = np.linalg.lstsq(A, ps[:, i], rcond=None)
            vel[i] = coef[1]
        return float(np.linalg.norm(vel))

    @property
    def veto_authority(self):
        """May project a tube in blocks(): confirmed AND the fitted
        speed is inside the layer's own mover envelope. A fabricated
        velocity from paired noise fails the envelope and gains no
        block authority at all (F2)."""
        if not self.confirmed:
            return False
        sp = self.fit_speed()
        return sp is not None and sp <= SPEED_ENVELOPE

    @property
    def full_authority(self):
        """May feed the dodge (the `confirmed` voxel set): veto
        authority PLUS a 3rd, motion-consistent appearance (F2)."""
        return (self.veto_authority and self.n_hits >= FULL_HITS
                and self.full_ok)

    def predict(self, t):
        pos, vel = self.estimate()
        if pos is None:
            return self.points[-1][1]
        return pos + vel * max(0.0, t - self.points[-1][0])

    def estimate(self):
        """(pos_at_latest_t, velocity) from the fit window, or
        (point, zero) with a single point; (None, None) if empty.
        Velocity is zeroed when the fitted speed exceeds SPEED_HI
        (association noise, not a mover)."""
        pts = self.points
        if not pts:
            return None, None
        if len(pts) == 1:
            return pts[0][1], np.zeros(3)
        ts = np.array([p[0] for p in pts])
        ps = np.array([p[1] for p in pts])
        if ts[-1] - ts[0] < 1e-9:
            return ps[-1], np.zeros(3)
        A = np.vstack([np.ones_like(ts), ts]).T
        vel = np.zeros(3)
        pos = np.zeros(3)
        for i in range(3):
            coef, *_ = np.linalg.lstsq(A, ps[:, i], rcond=None)
            vel[i] = coef[1]
            pos[i] = coef[0] + coef[1] * ts[-1]
        if float(np.linalg.norm(vel)) > SPEED_HI:
            return ps[-1].copy(), np.zeros(3)
        return pos, vel


class DynLayer:
    """Drop-in successor to motion.MotionBuffer (same update/blocks
    interface, plus the events/confirmed attributes dodge.observe
    consumes)."""

    def __init__(self, shape, gate_m=GATE_M, confirm_hits=CONFIRM_HITS,
                 track_ttl_s=TRACK_TTL_S, block_radius=BLOCK_RADIUS,
                 proj_horizon_s=PROJ_HORIZON_S):
        self.shape = tuple(int(s) for s in shape)
        self.gate_m = float(gate_m)
        self.confirm_hits = int(confirm_hits)
        self.track_ttl_s = float(track_ttl_s)
        self.block_radius = int(block_radius)
        self.proj_horizon_s = float(proj_horizon_s)
        self.prev_hits = None   # voxel -> point, previous scan
        self.prev_free = None   # voxel set, previous scan
        self.prev_appeared = set()  # appearance voxels, decayed (F3)
        self._appeared_t = {}   # voxel -> last t it appeared (decay)
        self.prev_wall_cands = []   # (t, centroid, voxels) suppressed
                                    # wall-adjacent clusters, last scan
        self.wall_chains = []   # F1 candidate chains: dicts with
                                # pts=[(t, centroid)], hot=bool
        self._hit_hist = []     # hit-voxel sets, last 3 scans (F1
                                # flicker filter: t-2 membership)
        self.tracks = []
        self.events = {}        # voxel -> last observed t (dodge compat)
        self.confirmed = set()  # observed voxels of FULL-authority
                                # tracks only (v3: dodge-facing set)
        self.t_now = 0.0
        self.n_events_total = 0     # lifetime appearance count (union)
        self.n_confirmed_total = 0  # lifetime confirmed-track count
        self.n_prev_events = 0      # channel (a) lifetime count
        self.n_map_events = 0       # channel (b) lifetime count
        self.n_wall_promoted = 0    # F1: voxels promoted from wall
                                    # candidates to appearances
        self.n_full_total = 0       # F2: lifetime full-authority count

    # -- ingestion ----------------------------------------------------
    def update(self, scans, t, logodds=None):
        """Ingest one 1 Hz scan."""
        self.t_now = float(t)
        hits, free = _scan_geometry(scans, self.shape)
        appeared = set()
        suppressed = set()  # qualified on a channel but killed ONLY
                            # by wall-adjacency suppression (F1)
        if self.prev_free is not None:
            raw = {v for v in hits if v in self.prev_free}
            prev_ch = {v for v in raw if not self._adjacent_prev_hit(v)}
            suppressed |= raw - prev_ch
            self.n_prev_events += len(prev_ch)
            appeared |= prev_ch
        if logodds is not None:
            raw = {v for v in hits if logodds[v] < FREE_THRESH}
            map_ch = {v for v in raw
                      if not self._adjacent_occupied(v, logodds)}
            suppressed |= raw - map_ch
            self.n_map_events += len(map_ch)
            appeared |= map_ch
        # F1: wall-adjacent candidates -> chains -> promotion.
        # Flicker filter first: a candidate voxel that was a HIT two
        # scans ago is a static voxel the sparse ray set repainted,
        # not a mover's leading edge — drop it from candidacy.
        cand_voxels = suppressed - appeared
        if len(self._hit_hist) >= 2:
            cand_voxels -= self._hit_hist[-2]
        cand_clusters = []
        for comp in _clusters(sorted(cand_voxels)):
            pts = np.array([hits[v] for v in comp])
            cand_clusters.append((pts.mean(axis=0), comp))
        promo = self._chain_candidates(cand_clusters)
        promoted_voxels = set()
        for ci in promo:
            promoted_voxels |= set(cand_clusters[ci][1])
        if promoted_voxels:
            appeared |= promoted_voxels
            self.n_wall_promoted += len(promoted_voxels)
        self.prev_wall_cands = [(self.t_now, cen, frozenset(comp))
                                for cen, comp in cand_clusters]
        self._hit_hist.append(set(hits))
        self._hit_hist = self._hit_hist[-3:]
        if appeared:
            self.n_events_total += len(appeared)
            cents = []
            for comp in _clusters(sorted(appeared)):
                pts = np.array([hits[v] for v in comp])
                cents.append((pts.mean(axis=0), comp))
            # promo keyed by candidate cluster -> re-key by appearance
            # cluster (a promoted cluster may merge with a passed one)
            promo_by_cent = {}
            if promo:
                for ci, (cen, comp) in enumerate(cents):
                    hit_prev = [promo[k] for k in promo
                                if set(comp) & set(cand_clusters[k][1])]
                    if hit_prev:
                        promo_by_cent[ci] = hit_prev[0]
            self._associate(cents, promo_by_cent)
        # F3: prev_appeared decays instead of hard replacement — one
        # zero-appearance scan must not empty it (self-lockout).
        for v in appeared:
            self._appeared_t[v] = self.t_now
        self._appeared_t = {v: tt for v, tt in self._appeared_t.items()
                            if self.t_now - tt <= APPEAR_DECAY_S}
        self.prev_appeared = set(self._appeared_t)
        self.prev_hits, self.prev_free = hits, free
        # expire stale tracks; rebuild the dodge-facing views
        self.tracks = [tr for tr in self.tracks
                       if self.t_now - tr.t_last <= self.track_ttl_s]
        self.events = {}
        self.confirmed = set()
        for tr in self.tracks:
            for v in tr.voxels:
                self.events[v] = tr.t_last
            if tr.full_authority:
                self.confirmed.update(tr.voxels)
                if not tr._full_counted:
                    tr._full_counted = True
                    self.n_full_total += 1

    def _chain_candidates(self, cand_clusters):
        """F1: advance wall-candidate chains by one scan; return
        promo = {cand_cluster_idx: (seed_points, consistent)} for
        clusters whose chain just proved motion consistency (3rd
        point within WALL_CHAIN_RESID_M of the CV prediction, steps
        in [WALL_MIN_MOVE_M, WALL_MAX_STEP_M], step direction cosine
        >= WALL_CHAIN_COS) or whose chain is already hot. Chains
        that miss a scan die — the evidence must be consecutive."""
        promo = {}
        pairs = []
        for hi, ch in enumerate(self.wall_chains):
            pts = ch["pts"]
            lt, lcen = pts[-1]
            if len(pts) >= 2:
                pt_, pcen_ = pts[-2]
                vel = (lcen - pcen_) / max(lt - pt_, 1e-9)
                pred = lcen + vel * (self.t_now - lt)
            else:
                pred = lcen
            for ci, (cen, _comp) in enumerate(cand_clusters):
                step = cen - lcen
                dstep = float(np.linalg.norm(step))
                if not (WALL_MIN_MOVE_M <= dstep <= WALL_MAX_STEP_M):
                    continue
                if len(pts) >= 2:
                    resid = float(np.linalg.norm(cen - pred))
                    if resid > WALL_CHAIN_RESID_M:
                        continue
                    if not ch["hot"]:
                        prev_step = pts[-1][1] - pts[-2][1]
                        pn = float(np.linalg.norm(prev_step))
                        if pn < 1e-9 or float(np.dot(step, prev_step)) \
                                / (dstep * pn) < WALL_CHAIN_COS:
                            continue
                    pairs.append((resid, hi, ci))
                else:
                    pairs.append((dstep, hi, ci))
        pairs.sort()
        used_h, used_c = set(), set()
        new_chains = []
        for _d, hi, ci in pairs:
            if hi in used_h or ci in used_c:
                continue
            used_h.add(hi)
            used_c.add(ci)
            ch = self.wall_chains[hi]
            ch["pts"].append((self.t_now, cand_clusters[ci][0]))
            ch["pts"] = ch["pts"][-3:]
            if ch["hot"] or len(ch["pts"]) >= 3:
                ch["hot"] = True
                promo[ci] = (list(ch["pts"][:-1]), True)
            new_chains.append(ch)
        for ci, (cen, _comp) in enumerate(cand_clusters):
            if ci not in used_c:
                new_chains.append({"pts": [(self.t_now, cen)],
                                   "hot": False})
        self.wall_chains = new_chains
        return promo

    def _adjacent_occupied(self, v, logodds):
        nx, ny, nz = self.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    w = (v[0] + dx, v[1] + dy, v[2] + dz)
                    if 0 <= w[0] < nx and 0 <= w[1] < ny and 0 <= w[2] < nz:
                        if logodds[w] > OCC_LO:
                            return True
        return False

    def _adjacent_prev_hit(self, v):
        """Adjacent to a previous-scan hit that was NOT itself an
        appearance (i.e. static structure). A mover's own previous
        position must not suppress its next step."""
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    w = (v[0] + dx, v[1] + dy, v[2] + dz)
                    if w in self.prev_hits and w not in self.prev_appeared:
                        return True
        return False

    def _split_contended(self, cents, promo):
        """F3: a cluster contended by >= 2 tracks with separated
        predictions (>= 0.75 m apart) is two movers merged by
        26-connectivity, not one object. Split its voxels by nearest
        prediction so each track keeps its own centroid/velocity."""
        if len(self.tracks) < 2:
            return cents, promo
        preds = [tr.predict(self.t_now) for tr in self.tracks]
        out, out_promo = [], {}
        for ci, (cen, comp) in enumerate(cents):
            cont = [ti for ti, p in enumerate(preds)
                    if float(np.linalg.norm(cen - p)) <= self.gate_m]
            if len(cont) < 2 or len(comp) < 2:
                out.append((cen, comp))
                if ci in promo:
                    out_promo[len(out) - 1] = promo[ci]
                continue
            ps = [preds[ti] for ti in cont]
            sep = max(float(np.linalg.norm(ps[i] - ps[j]))
                      for i in range(len(ps)) for j in range(i + 1, len(ps)))
            if sep < 0.75:
                out.append((cen, comp))
                if ci in promo:
                    out_promo[len(out) - 1] = promo[ci]
                continue
            # reserve each contending track's nearest voxel first, so
            # no contender is left with an empty piece
            remaining = list(comp)
            pieces = {ti: [] for ti in cont}
            for ti in sorted(cont, key=lambda k: float(
                    np.linalg.norm(cen - preds[k]))):
                if not remaining:
                    break
                v = min(remaining, key=lambda w: float(np.linalg.norm(
                    np.array([w[0] + 0.5, w[1] + 0.5, w[2] + 0.5])
                    - preds[ti])))
                pieces[ti].append(v)
                remaining.remove(v)
            for w in remaining:
                ctr = np.array([w[0] + 0.5, w[1] + 0.5, w[2] + 0.5])
                ti = min(cont, key=lambda k: float(
                    np.linalg.norm(ctr - preds[k])))
                pieces[ti].append(w)
            made = [p for p in pieces.values() if p]
            if len(made) < 2:
                out.append((cen, comp))
                if ci in promo:
                    out_promo[len(out) - 1] = promo[ci]
                continue
            # piece centroids in the hit-point frame: voxel-center
            # mean plus the parent cluster's sub-voxel offset (the
            # difference between its hit-point centroid and its own
            # voxel-center mean), so splitting adds no frame jump
            delta = cen - np.mean([[w[0] + 0.5, w[1] + 0.5, w[2] + 0.5]
                                   for w in comp], axis=0)
            for piece in made:
                ctr = np.mean([[w[0] + 0.5, w[1] + 0.5, w[2] + 0.5]
                               for w in piece], axis=0) + delta
                out.append((ctr, piece))
                if ci in promo:
                    # the promo (wall-chain seed points) belongs to
                    # the piece nearest the chain's latest point
                    last_seed = promo[ci][0][-1][1]
                    if float(np.linalg.norm(ctr - last_seed)) <= 1.0:
                        out_promo[len(out) - 1] = promo[ci]
        return out, out_promo

    def _associate(self, cents, promo=None):
        promo = promo or {}
        cents, promo = self._split_contended(cents, promo)
        # greedy one-to-one: nearest (track, cluster) pairs first
        pairs = []
        for ti, tr in enumerate(self.tracks):
            pred = tr.predict(self.t_now)
            for ci, (cen, _comp) in enumerate(cents):
                d = float(np.linalg.norm(cen - pred))
                if d <= self.gate_m:
                    pairs.append((d, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            tr = self.tracks[ti]
            cen, comp = cents[ci]
            was = tr.confirmed
            # F1 promo: seed the wall-chain points first (unless the
            # track already has points at those times — a continuing
            # wall-hugger's recent points ARE the chain points).
            if ci in promo:
                seeds, _cons = promo[ci]
                for pt, pcen in seeds:
                    if all(abs(tt - pt) > 0.01 for tt, _ in tr.points):
                        tr.points.append((pt, pcen))
                        tr.n_hits += 1
            tr.points.append((self.t_now, cen))
            tr.points = sorted(
                [(tt, pp) for (tt, pp) in tr.points
                 if tt >= self.t_now - FIT_WINDOW_S])
            tr.voxels = list(comp)
            tr.n_hits += 1
            tr.t_last = self.t_now
            # F2: the appearance that takes a track to >= FULL_HITS
            # grants full authority only if it was motion-consistent
            if tr.n_hits >= FULL_HITS and d <= FULL_GATE_M:
                tr.full_ok = True
            if tr.confirmed and not was:
                self.n_confirmed_total += 1
        for ci, (cen, comp) in enumerate(cents):
            if ci not in used_c:
                tr = _Track(self.t_now, cen, comp)
                if ci in promo:
                    seeds, _consistent = promo[ci]
                    tr.points = list(seeds) + [(self.t_now, cen)]
                    tr.n_hits = len(tr.points)
                    if tr.confirmed:
                        self.n_confirmed_total += 1
                    # NOTE: chain promotion grants VETO authority
                    # only. Full (dodge) authority still requires
                    # one more associated appearance landing within
                    # FULL_GATE_M of the seeded track's prediction
                    # (the ordinary F2 rule in the loop above) —
                    # measured reason: flicker chains do occur in
                    # candidate-dense rooms (071 baseline), and a
                    # 4th consistent point is what separates them
                    # from a persisting wall-hugger.
                self.tracks.append(tr)

    # -- queries --------------------------------------------------------
    def blocks(self, voxel):
        """True if `voxel` is within block_radius of any veto-authority
        track's projected tube over [now, now + PROJ_HORIZON]."""
        if not self.tracks:
            return False
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        r = self.block_radius
        n_steps = int(round(self.proj_horizon_s / PROJ_STEP_S))
        for tr in self.tracks:
            if not tr.veto_authority:
                continue
            pos, vel = tr.estimate()
            if pos is None:
                continue
            for k in range(n_steps + 1):
                c = pos + vel * (k * PROJ_STEP_S)
                # F4: compare against the voxel CENTER (v + 0.5) with
                # radius r + 0.5 — the same Chebyshev coverage as the
                # old |int(c) - v| <= r test, but continuous: the old
                # int() truncation turned a fit value of 9.9999999
                # (true position 10.0) into voxel 9, shifting the
                # tube up to 1 voxel toward -x/-y asymmetrically.
                lim = r + 0.5 + 1e-6  # 1e-6: fit float slop at the
                                      # exact boundary (9.9999999 case)
                if abs(c[0] - (vx + 0.5)) <= lim \
                        and abs(c[1] - (vy + 0.5)) <= lim \
                        and abs(c[2] - (vz + 0.5)) <= lim:
                    return True
        return False

    @property
    def n_active(self):
        return len(self.events)

    @property
    def n_tracks_confirmed(self):
        return sum(1 for tr in self.tracks if tr.confirmed)
