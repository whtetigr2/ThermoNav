"""Unit smoke tests for dynlayer.DynLayer (synthetic scans, no sim).

Run: python3 test_dynlayer.py  -> prints PASS lines, asserts on failure.
"""

import numpy as np

import dynlayer

SHAPE = (30, 11, 11)
ORIGIN_A = (0.5, 5.5, 5.5)   # ray along +x at y=5
ORIGIN_B = (0.5, 6.5, 5.5)   # ray along +x at y=6
DIR_X = (1.0, 0.0, 0.0)


def ray(origin, meas):
    return (origin, DIR_X, meas, True, True)


def wall_scans():
    return [ray(ORIGIN_A, 19.5), ray(ORIGIN_B, 19.5)]


def test_static_no_events():
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    for t in range(5):
        dl.update(wall_scans(), float(t), lo)
    assert dl.n_events_total == 0, dl.n_events_total
    assert not dl.tracks
    assert not dl.blocks((10, 5, 5))
    print("PASS static: no events, no blocks")


def test_noise_short_of_wall_suppressed():
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    dl.update(wall_scans(), 0.0, lo)
    noisy = [ray(ORIGIN_A, 18.5), ray(ORIGIN_B, 19.5)]  # 1 voxel short
    dl.update(noisy, 1.0, lo)
    assert dl.n_events_total == 0, dl.n_events_total
    print("PASS noise: short-of-wall hit suppressed")


def test_crossing_mover_confirms_and_projects():
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    dl.update(wall_scans(), 0.0, lo)
    # t=1: mover at (10,5,5) crosses ray A
    dl.update([ray(ORIGIN_A, 9.5), ray(ORIGIN_B, 19.5)], 1.0, lo)
    assert dl.n_events_total == 1, dl.n_events_total
    assert len(dl.tracks) == 1 and not dl.tracks[0].confirmed
    # t=2: mover stepped +y to (10,6,5), now on ray B
    dl.update([ray(ORIGIN_A, 19.5), ray(ORIGIN_B, 9.5)], 2.0, lo)
    assert len(dl.tracks) == 1, len(dl.tracks)
    tr = dl.tracks[0]
    assert tr.confirmed, "track must confirm on 2nd appearance"
    pos, vel = tr.estimate()
    assert abs(vel[1] - 1.0) < 0.01, vel  # +y at 1 m/s
    # projected tube: 1.5 s ahead the mover is at y~8 -> blocked
    assert dl.blocks((10, 8, 5)), "projected voxel must block"
    # behind the mover: not blocked
    assert not dl.blocks((10, 3, 5))
    # far away: not blocked
    assert not dl.blocks((25, 5, 5))
    print("PASS crossing: confirmed at scan 2, velocity +y 1.0, "
          "projection blocks ahead only")


def test_headon_velocity():
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    dl.update(wall_scans(), 0.0, lo)
    dl.update([ray(ORIGIN_A, 14.5), ray(ORIGIN_B, 19.5)], 1.0, lo)
    dl.update([ray(ORIGIN_A, 13.5), ray(ORIGIN_B, 19.5)], 2.0, lo)
    assert len(dl.tracks) == 1 and dl.tracks[0].confirmed
    _pos, vel = dl.tracks[0].estimate()
    assert abs(vel[0] + 1.0) < 0.01, vel  # approaching at 1 m/s
    print("PASS head-on: velocity -x 1.0 at confirmation")


def test_track_expires():
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    dl.update(wall_scans(), 0.0, lo)
    dl.update([ray(ORIGIN_A, 9.5), ray(ORIGIN_B, 19.5)], 1.0, lo)
    dl.update([ray(ORIGIN_A, 19.5), ray(ORIGIN_B, 9.5)], 2.0, lo)
    assert dl.blocks((10, 8, 5))
    for t in (3.0, 4.0, 5.0, 6.0):
        dl.update(wall_scans(), t, lo)
    assert not dl.tracks, "stale track must expire"
    assert not dl.blocks((10, 8, 5))
    print("PASS expiry: track dropped after TTL, no stale blockade")


def _ray_y(y, meas):
    return ((0.5, float(y), 5.5), DIR_X, float(meas), True, True)


def _wall_scans_y(ys=range(1, 11)):
    return [_ray_y(y, 19.5) for y in ys]


def _scans_y(hit_map, ys=range(1, 11)):
    return [_ray_y(y, hit_map.get(y, 19.5)) for y in ys]


def test_wall_hugging_mover_promotes():
    """F1 (red-team U2): a mover 1 voxel in front of a wall, walking
    along it, was invisible to BOTH channels. Now candidates chain:
    scans 1-2 silent (chain building — a 2-scan movement test was
    measured to promote wall flicker in closed loop), scan 3 the
    chain proves motion consistency and promotes, seeding the track
    with the chain points (real velocity, veto authority); scan 4
    (hot chain) supplies the 4th consistent point -> full."""
    shape = (30, 12, 11)
    dl = dynlayer.DynLayer(shape)
    lo = np.full(shape, -2.0)
    lo[20, :, :] = 2.0
    dl.update(_wall_scans_y(), 0.0, lo)
    dl.update(_scans_y({3: 18.5}), 1.0, lo)  # mover at (19,3,5)
    assert dl.n_events_total == 0, dl.n_events_total  # candidate only
    dl.update(_scans_y({4: 18.5}), 2.0, lo)
    assert dl.n_events_total == 0, dl.n_events_total  # chain, 2 pts
    assert not dl.tracks
    dl.update(_scans_y({5: 18.5}), 3.0, lo)
    assert dl.n_wall_promoted >= 1, dl.n_wall_promoted
    assert len(dl.tracks) == 1 and dl.tracks[0].veto_authority
    _pos, vel = dl.tracks[0].estimate()
    assert abs(vel[1] - 1.0) < 0.01, vel  # real +y velocity, seeded
    assert dl.blocks((19, 5, 5)), "promoted track must veto"
    assert not dl.tracks[0].full_authority, \
        "chain promotion is veto-only; full needs a 4th point"
    assert not dl.confirmed
    dl.update(_scans_y({6: 18.5}), 4.0, lo)  # hot chain: stays live
    assert dl.tracks[0].n_hits >= 4, dl.tracks[0].n_hits
    assert dl.tracks[0].full_authority, "4th consistent point -> full"
    assert (19, 6, 5) in dl.confirmed
    print("PASS wall-hug: chain promotes at scan 3 (veto), full at 4")


def test_stationary_wall_candidate_never_promotes():
    """F1 guard: a wall-adjacent hit that does NOT move is wall noise,
    however often it repeats."""
    shape = (30, 12, 11)
    dl = dynlayer.DynLayer(shape)
    lo = np.full(shape, -2.0)
    lo[20, :, :] = 2.0
    dl.update(_wall_scans_y(), 0.0, lo)
    for t in (1.0, 2.0, 3.0):
        dl.update(_scans_y({5: 18.5}), t, lo)
    assert dl.n_events_total == 0, dl.n_events_total
    assert not dl.tracks
    print("PASS wall-noise stationary: never promoted, stays silent")


def test_phantom_velocity_no_authority():
    """F2 (red-team U4): two unrelated noise hits inside the gate
    fabricate (+2.0,+2.0) m/s. The pair still forms a track, but the
    fitted speed is outside the mover envelope: no tube, no dodge."""
    shape = (30, 12, 11)
    dl = dynlayer.DynLayer(shape)
    lo = np.zeros(shape)
    dl.update(_wall_scans_y(), 0.0, lo)
    dl.update(_scans_y({3: 9.5}), 1.0, lo)    # noise at (10,3)
    dl.update(_scans_y({5: 11.5}), 2.0, lo)   # noise at (12,5)
    assert len(dl.tracks) == 1 and dl.tracks[0].confirmed  # 2 hits
    assert not dl.tracks[0].veto_authority, "2.83 m/s fit must not veto"
    assert not dl.blocks((12, 5, 5))
    assert not dl.blocks((15, 9, 5)), "no fabricated-velocity tube"
    assert not dl.confirmed, "no dodge authority"
    print("PASS phantom velocity: track exists, zero authority")


def test_full_authority_needs_third_consistent_hit():
    """F2: head-on mover — veto at hit 2, dodge-facing confirmed set
    only at hit 3."""
    dl = dynlayer.DynLayer(SHAPE)
    lo = np.zeros(SHAPE)
    dl.update(wall_scans(), 0.0, lo)
    dl.update([ray(ORIGIN_A, 14.5), ray(ORIGIN_B, 19.5)], 1.0, lo)
    dl.update([ray(ORIGIN_A, 13.5), ray(ORIGIN_B, 19.5)], 2.0, lo)
    assert dl.tracks[0].veto_authority and not dl.confirmed
    assert dl.blocks((14, 5, 5)), "veto tube live at hit 2"
    dl.update([ray(ORIGIN_A, 12.5), ray(ORIGIN_B, 19.5)], 3.0, lo)
    assert dl.tracks[0].full_authority
    assert (13, 5, 5) in dl.confirmed
    print("PASS tiered authority: hit2 veto-only, hit3 full")


def test_two_movers_crossing_split_and_decay():
    """F3 (red-team U1): two movers crossing merge into one cluster
    at the crossing scan. Splitting keeps both identities/velocities;
    prev_appeared decay prevents self-lockout after the swap scan
    (t=4 produces no appearances — the movers swap voxels in place,
    a documented differencing residual — and t=5 must still fire)."""
    shape = (30, 12, 11)
    dl = dynlayer.DynLayer(shape)
    lo = np.zeros(shape)
    dl.update(_wall_scans_y(), 0.0, lo)
    seq = {1: {3: 9.5, 8: 9.5}, 2: {4: 9.5, 7: 9.5}, 3: {5: 9.5, 6: 9.5},
           4: {6: 9.5, 5: 9.5}, 5: {7: 9.5, 4: 9.5}}
    ev = {}
    for t in range(1, 6):
        before = dl.n_events_total
        dl.update(_scans_y(seq[t]), float(t), lo)
        ev[t] = dl.n_events_total - before
    assert ev[3] == 2, ev  # merged cluster split into 2 appearances
    assert ev[4] == 0, ev  # swap-in-place: no differencing evidence
    assert ev[5] == 2, ev  # decay: own trail does not suppress return
    vels = sorted(float(tr.estimate()[1][1])
                  for tr in dl.tracks if tr.confirmed)
    # split pieces carry voxel-center centroids, so allow quantization
    # slop; the red-team failure was ONE track at -1.25 (identity lost)
    assert len(vels) == 2 and abs(vels[0] + 1.0) < 0.1 \
        and abs(vels[1] - 1.0) < 0.1, vels
    print("PASS two-mover: split at merge, no lockout, velocities "
          "+1.0/-1.0 recovered")


def test_tube_rounding_symmetric():
    """F4: a fitted position of 9.9999999 must center the tube on
    voxel 10, not truncate to 9 (coverage symmetric +-1)."""
    dl = dynlayer.DynLayer(SHAPE)
    tr = dynlayer._Track(0.0, np.array([10.0, 5.5, 5.5]), [(10, 5, 5)])
    tr.points = [(0.0, np.array([10.0, 5.5, 5.5])),
                 (1.0, np.array([9.9999999, 5.5, 5.5]))]
    tr.n_hits = 2
    tr.t_last = 1.0
    dl.tracks = [tr]
    dl.t_now = 1.0
    for v in ((10, 5, 5), (11, 5, 5), (9, 5, 5), (10, 6, 5), (10, 4, 5)):
        assert dl.blocks(v), v
    assert not dl.blocks((12, 5, 5))
    print("PASS tube rounding: symmetric coverage around voxel 10")


if __name__ == "__main__":
    test_static_no_events()
    test_noise_short_of_wall_suppressed()
    test_crossing_mover_confirms_and_projects()
    test_headon_velocity()
    test_track_expires()
    test_wall_hugging_mover_promotes()
    test_stationary_wall_candidate_never_promotes()
    test_phantom_velocity_no_authority()
    test_full_authority_needs_third_consistent_hit()
    test_two_movers_crossing_split_and_decay()
    test_tube_rounding_symmetric()
    print("ALL PASS")
