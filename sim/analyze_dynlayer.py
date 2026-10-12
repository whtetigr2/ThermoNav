"""Paired analysis for dynlayer_sweep.csv: OFF vs ON, same cells/seeds.

Also places the ON arm next to the prior published arms (dynamic_sweep
43/48 no layer, motion_sweep 30/48, dodge_sweep 26/48) — those were run
at their own commits on the same protocol; cited as context, not as
paired data.
"""

import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


def coll(rows):
    return sum(int(r["collision"]) for r in rows), len(rows)


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: coll(v) for k, v in sorted(groups.items())}


def fmt(pair):
    return f"{pair[0]}/{pair[1]}"


def main():
    rows = load("dynlayer_sweep.csv")
    movers = [r for r in rows if r["cell"] != "baseline"]
    base = [r for r in rows if r["cell"] == "baseline"]
    off = [r for r in movers if r["arm"] == "off"]
    on = [r for r in movers if r["arm"] == "on"]

    print("== Primary: closed-loop mover collisions (48 paired cells) ==")
    print(f"OFF (no layer):        {fmt(coll(off))}")
    print(f"ON  (dynlayer+dodge):  {fmt(coll(on))}")
    for prior, name in (("dynamic_sweep.csv", "prior no-layer (2026-10-10)"),
                        ("motion_sweep.csv", "prior motion-only"),
                        ("dodge_sweep.csv", "prior motion+dodge")):
        try:
            pr = [r for r in load(prior) if r["cell"] != "baseline"]
            print(f"context {name}: {fmt(coll(pr))} (own commit, unpaired)")
        except FileNotFoundError:
            pass

    # paired cell-by-cell flips
    off_by = {r["mission_id"].replace("dyn2_off_", ""): r for r in off}
    on_by = {r["mission_id"].replace("dyn2_on_", ""): r for r in on}
    fixed, broken = [], []
    for k in sorted(off_by):
        if k not in on_by:
            continue
        oc, nc = int(off_by[k]["collision"]), int(on_by[k]["collision"])
        if oc and not nc:
            fixed.append(k)
        elif nc and not oc:
            broken.append(k)
    print(f"\npaired flips: fixed by ON: {len(fixed)}, broken by ON: {len(broken)}")
    for k in fixed:
        print(f"  fixed: {k}")
    for k in broken:
        print(f"  BROKEN: {k}")

    for key in ("mover_geometry", "mover_speed", "room"):
        print(f"\nby {key}:  OFF -> ON")
        bo, bn = breakdown(off, key), breakdown(on, key)
        for k in bo:
            print(f"  {k}: {fmt(bo[k])} -> {fmt(bn.get(k, (0, 0)))}")

    print("\n== Static regression + false-positive cost (4 baselines/arm) ==")
    for arm in ("off", "on"):
        bs = [r for r in base if r["arm"] == arm]
        for r in bs:
            print(f"  {arm} {r['room']} s{r['seed']}: {r['outcome']} "
                  f"events={r['n_motion_events']} "
                  f"mref={r['n_motion_refusals']} "
                  f"dodges={r['n_dodges']} "
                  f"path_eff={r['path_efficiency']}")

    print("\n== ON-arm diagnostics (mover cells) ==")
    if on:
        n = len(on)
        print(f"  mean appearances/mission: "
              f"{sum(int(r['n_motion_events']) for r in on) / n:.1f} "
              f"(prev-scan ch: "
              f"{sum(int(r['n_dyn_prev_events']) for r in on) / n:.1f}, "
              f"map ch: {sum(int(r['n_dyn_map_events']) for r in on) / n:.1f})")
        print(f"  missions with >=1 confirmed track: "
              f"{sum(1 for r in on if int(r['n_motion_confirmed']) > 0)}/{n}")
        print(f"  mean motion refusals/mission: "
              f"{sum(int(r['n_motion_refusals']) for r in on) / n:.1f}")
        print(f"  total dodges: {sum(int(r['n_dodges']) for r in on)}, "
              f"fallbacks: {sum(int(r['n_dodge_fallbacks']) for r in on)}, "
              f"wall incidents: "
              f"{sum(int(r['n_dodge_wall_incidents']) for r in on)}")
        succ_on = [r for r in on if not int(r["collision"])]
        print(f"  ON successes: {len(succ_on)}/{n}")


if __name__ == "__main__":
    main()
