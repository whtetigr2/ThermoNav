"""Control-link health model for the S16 dynamics stack (LEGO, 2026-10-11).

Roadmap Phase 1, item 2: the escalation ladder's link trigger slot was a
MISSING stub ("no comms model in sim"). This module is that comms model —
the smallest honest version:

  LinkChannel     — the wire. GCS-origin packets (heartbeat / operator
                    intent / override) are sent at PACKET_RATE_HZ, delayed
                    by a sampled latency, and dropped with the condition's
                    loss probability. Two channel drivers:
                      * "scripted": condition schedule [(t0, t1, cond), ...]
                        — deterministic scenario levers for validation.
                      * "markov":   a two-state Gilbert-Elliott style chain
                        (good/degraded) + Poisson dropout bursts — the
                        stochastic model for sweeps.
                    The hidden condition / chain state lives ONLY here.
  LinkHealthTracker — the receiver. Sees ONLY what a real receiver can
                    report: delivered-packet timestamps, per-packet
                    latency (from packet timestamps), and expected-vs-
                    delivered counts in a trailing window. (Simplification,
                    flagged by red-team round 1: the channel feeds the
                    tracker expected counts from its own tick accounting;
                    a real receiver would infer gaps from sequence
                    numbers. The observability boundary — tracker never
                    sees the hidden condition — is unaffected.) It never
                    sees the channel condition.
                    Output: HEALTHY / DEGRADED / LOST + time-in-state,
                    trailing loss %, latency EWMA, silence duration.

NO MEASURED LINK DATA EXISTS in this lab. Every parameter below is
[cited] from published autopilot practice or [ASSUMED] as a stress /
nominal value and labeled as such. Nothing here is a hardware claim.
Claim label: host-side classical model, validated in simulation only.

Parameter provenance
--------------------
PACKET_RATE_HZ = 10 Hz        [ASSUMED] supervised command/heartbeat
                               stream rate for this stack (MAVLink
                               heartbeats are commonly 1 Hz; command
                               streams run faster — 10 Hz matches our
                               10 Hz decision substep, so one packet per
                               substep; a modeling choice, not a radio
                               spec).
LOST_SILENCE_S = 5.0 s        [cited — value only] ArduPilot GCS failsafe:
                               triggers when no MAVLink heartbeat is
                               received for FS_GCS_TIMEOUT seconds,
                               default 5 s (ardupilot.org/copter/docs/
                               gcs-failsafe.html, read 2026-10-11).
                               DIVERGENCE (red-team F6): ArduPilot keeps
                               the failsafe INACTIVE if no GCS was ever
                               connected; this tracker declares LOST 5 s
                               after start if no packet ever arrives.
                               In-sim the GCS always sends, so the
                               divergence is latent, not active.
Conditions (per-packet):
  good:     latency 25 +/- 8 ms, loss 0.5 %
                               [ASSUMED] nominal short-range telemetry;
                               order-of-magnitude consistent with
                               published C2 targets (control latency
                               <20 ms desired, packet loss <1 % —
                               GTU Communication & Navigation lecture
                               notes, Unit 4, read 2026-10-11). Not
                               measured on any radio we own.
  degraded: latency 180 +/- 60 ms, loss 25 %
                               [ASSUMED] stress condition: an order of
                               magnitude worse than nominal on both axes.
  dropout:  loss 100 %         [ASSUMED] definition of a dropout burst.
Markov chain (per-second transition probabilities, converted per tick):
  p(good->degraded) = 0.02 /s  [ASSUMED]
  p(degraded->good) = 0.20 /s  [ASSUMED] mean degraded dwell ~5 s
  dropout bursts: rate 0.01 /s, duration lognormal median 1.5 s
                               [ASSUMED] stress process for sweeps; the
                               validation scenarios use the scripted
                               driver, not these numbers.
Health classification thresholds:
  DEGRADED if trailing-2 s loss >= 20 % OR latency EWMA >= 150 ms
                               [ASSUMED] receiver-side thresholds; 20 %
                               sits well below the degraded condition's
                               25 % mean and well above nominal 0.5 %.
  LOST if silence >= LOST_SILENCE_S (5.0 s)  [cited, above]
"""

import numpy as np
from collections import deque

HEALTHY, DEGRADED, LOST = 0, 1, 2
HEALTH_NAMES = ["HEALTHY", "DEGRADED", "LOST"]

PACKET_RATE_HZ = 10.0        # [ASSUMED] see module docstring
LOST_SILENCE_S = 5.0         # [cited] ArduPilot FS_GCS_TIMEOUT default
WINDOW_S = 2.0               # [ASSUMED] trailing loss/latency window
DEGRADED_LOSS_FRAC = 0.20    # [ASSUMED]
DEGRADED_LATENCY_MS = 150.0  # [ASSUMED]
HYST_LATENCY_MS = 100.0      # [ASSUMED] DEGRADED exit needs EWMA below this
HYST_LOSS_FRAC = 0.10        # [ASSUMED] ... and window loss below this

# condition -> (latency mean ms, latency sd ms, loss probability)
CONDITIONS = {
    "good":     (25.0, 8.0, 0.005),    # [ASSUMED] nominal
    "degraded": (180.0, 60.0, 0.25),   # [ASSUMED] stress
    "dropout":  (0.0, 0.0, 1.0),       # [ASSUMED] total loss
}

PACKET_TTL_S = 2.0           # [design] packets whose sampled latency
                             # exceeds this are discarded as stale at
                             # send time (counted in stats["expired"]),
                             # so a late operator override cannot act
                             # unboundedly late (red-team F7). With the
                             # condition parameters below, sampled
                             # latencies stay <~0.5 s, so this binds only
                             # if the parameters change.

# Markov driver [all ASSUMED]
P_GOOD_TO_DEG_PER_S = 0.02
P_DEG_TO_GOOD_PER_S = 0.20
DROPOUT_RATE_PER_S = 0.01
DROPOUT_MEDIAN_S = 1.5


class LinkHealthTracker:
    """Receiver-side health estimator. Observable inputs ONLY:
    delivery events (time, latency) and expected-packet counts.
    The channel's hidden condition is never passed in."""

    def __init__(self, t_start=0.0):
        self.t_start = float(t_start)
        self.last_delivery_t = None
        self.latency_ewma_ms = None
        # trailing window of (t, expected, delivered) buckets
        self._buckets = deque()
        self._exp_total = 0
        self._del_total = 0
        self.state = HEALTHY
        self.state_since = float(t_start)
        self.n_delivered = 0
        self.n_expected = 0

    # -- observable inputs --------------------------------------------------
    def note_window(self, t, n_expected, latencies_ms):
        """Record one accounting window (e.g. one 0.1 s packet tick):
        how many packets were expected and the latencies of those that
        arrived (delivered count = len(latencies_ms))."""
        lat = list(latencies_ms)
        self._buckets.append((float(t), int(n_expected), len(lat)))
        self._exp_total += int(n_expected)
        self._del_total += len(lat)
        self.n_expected += int(n_expected)
        self.n_delivered += len(lat)
        for L in lat:
            self.last_delivery_t = float(t)
            self.latency_ewma_ms = L if self.latency_ewma_ms is None \
                else 0.7 * self.latency_ewma_ms + 0.3 * L
        # prune buckets outside the trailing window
        while self._buckets and self._buckets[0][0] < t - WINDOW_S:
            _, e0, d0 = self._buckets.popleft()
            self._exp_total -= e0
            self._del_total -= d0

    def window_loss_frac(self):
        # Clamped to [0, 1]: arrivals are credited to the arrival tick
        # while expected counts accrue per send tick, so a burst of
        # late arrivals could otherwise push the raw fraction negative
        # (red-team F7; observed in 2/3600 sampled degraded reports).
        if self._exp_total <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - self._del_total / self._exp_total))

    def silence_s(self, t):
        ref = self.last_delivery_t if self.last_delivery_t is not None \
            else self.t_start
        return max(0.0, float(t) - ref)

    # -- health output --------------------------------------------------------
    def report(self, t):
        """Health report at time t. Returns a plain dict — this, and only
        this, is what the escalation policy is allowed to consume."""
        t = float(t)
        silence = self.silence_s(t)
        loss = self.window_loss_frac()
        ewma = self.latency_ewma_ms if self.latency_ewma_ms is not None \
            else 0.0
        if silence >= LOST_SILENCE_S:
            new = LOST
        elif self.state == LOST:
            new = DEGRADED  # first packet back: recovering, not yet healthy
        elif loss >= DEGRADED_LOSS_FRAC or ewma >= DEGRADED_LATENCY_MS:
            new = DEGRADED
        elif self.state == DEGRADED and (
                loss >= HYST_LOSS_FRAC or ewma >= HYST_LATENCY_MS):
            new = DEGRADED  # hysteresis: stay until clearly recovered
        else:
            new = HEALTHY
        if new != self.state:
            self.state = new
            self.state_since = t
        # continuous score [design]: 1 = perfect, 0 = lost (clamped)
        score = 0.0 if new == LOST else min(1.0, max(
            0.0, 1.0 - 0.7 * loss - 0.3 * min(ewma / 300.0, 1.0)))
        if new == DEGRADED:
            score = min(score, 0.6)
        return {
            "state": new,
            "state_name": HEALTH_NAMES[new],
            "score": round(float(score), 3),
            "time_in_state_s": round(t - self.state_since, 3),
            "silence_s": round(silence, 3),
            "window_loss_frac": round(float(loss), 3),
            "latency_ewma_ms": round(float(ewma), 1),
            "n_delivered": self.n_delivered,
            "n_expected": self.n_expected,
        }


class LinkChannel:
    """The wire between GCS and vehicle (uplink: heartbeat / intent /
    operator override packets).

    send(t, payload): one packet enters the wire. Depending on the hidden
        condition it is dropped (loss) or queued with arrival = t + latency.
    poll(t): returns payloads whose arrival time has come (and advances
        the receiver tracker's accounting for the elapsed ticks).

    Accounting model: the sender emits exactly one packet per tick
    (dt = 1/PACKET_RATE_HZ). The tracker is credited per tick with
    expected=1 and the latencies of packets that ARRIVE during that tick
    (a real receiver counts arrivals and infers gaps from sequence
    numbers; here the channel's tick accounting feeds the expected
    counts directly — the simplification in the module docstring).
    Dropped packets are visible only as missing expected counts in the
    window and as growing silence.
    """

    def __init__(self, spec=None, seed=0):
        spec = dict(spec or {})
        self.mode = spec.get("mode", "nominal")
        self.schedule = [tuple(x) for x in spec.get("schedule", [])]
        self.rng = np.random.default_rng(seed)
        self.tracker = LinkHealthTracker()
        self._queue = []  # (arrival_t, seq, payload)
        self._seq = 0
        self._next_tick = 0.0
        # markov hidden state
        self._mk_state = "good"
        self._mk_dropout_until = -1.0
        self.stats = {"sent": 0, "dropped": 0, "delivered": 0,
                      "expired": 0, "max_latency_ms": 0.0,
                      "dropout_ticks": 0}

    # -- hidden condition (NEVER exposed to tracker/policy) -----------------
    def _condition_at(self, t):
        if self.mode == "nominal":
            return "good"
        if self.mode == "scripted":
            for t0, t1, cond in self.schedule:
                if t0 <= t < t1:
                    return cond
            return "good"
        if self.mode == "markov":
            if t < self._mk_dropout_until:
                return "dropout"
            return "degraded" if self._mk_state == "degraded" else "good"
        raise ValueError(f"unknown link mode {self.mode}")

    def _advance_markov(self, dt):
        if self.mode != "markov":
            return
        r = self.rng.random()
        if self._mk_state == "good":
            if r < P_GOOD_TO_DEG_PER_S * dt:
                self._mk_state = "degraded"
        else:
            if r < P_DEG_TO_GOOD_PER_S * dt:
                self._mk_state = "good"
        if self.rng.random() < DROPOUT_RATE_PER_S * dt:
            dur = float(self.rng.lognormal(
                mean=np.log(DROPOUT_MEDIAN_S), sigma=0.5))
            self._mk_dropout_until = max(
                self._mk_dropout_until, self._next_tick + dur)

    # -- wire -----------------------------------------------------------------
    def send(self, t, payload=None):
        """Emit one packet at time t under the current hidden condition."""
        cond = self._condition_at(float(t))
        mean_ms, sd_ms, p_loss = CONDITIONS[cond]
        self.stats["sent"] += 1
        if cond == "dropout":
            self.stats["dropout_ticks"] += 1
        if self.rng.random() < p_loss:
            self.stats["dropped"] += 1
            return False
        lat_ms = max(1.0, float(self.rng.normal(mean_ms, sd_ms)))
        if lat_ms > PACKET_TTL_S * 1000.0:
            self.stats["expired"] += 1  # stale at send: never queued
            return False
        self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"],
                                           lat_ms)
        # _seq stamps queued packets only (drops consume no sequence
        # number — the tracker is fed expected counts per tick instead;
        # see the module docstring simplification note).
        self._queue.append((float(t) + lat_ms / 1000.0, self._seq,
                            payload or {}, lat_ms))
        self._seq += 1
        return True

    def step(self, t_end):
        """Advance the wire to t_end in packet ticks, feeding the tracker.
        Returns payloads delivered up to t_end (in arrival order)."""
        delivered = []
        dt = 1.0 / PACKET_RATE_HZ
        while self._next_tick < float(t_end) - 1e-9:
            t_tick = self._next_tick
            self._advance_markov(dt)
            still, lats = [], []
            for arr, seq, pl, lat_ms in self._queue:
                if arr <= t_tick + dt + 1e-9:
                    delivered.append(pl)
                    self.stats["delivered"] += 1
                    lats.append(lat_ms)
                else:
                    still.append((arr, seq, pl, lat_ms))
            self._queue = still
            self.tracker.note_window(t_tick + dt, 1, lats)
            self._next_tick = t_tick + dt
        return delivered
