"""Executed-tree provenance guard (red-team round 1, F1; drift doctrine:
fix the system, not just the number).

Round-1 failure: harnesses recorded `git rev-parse HEAD` captured at
script launch, in a workflow where the tree was uncommitted (calibration
CSV claimed a commit where the code did not exist) or landed mid-run
(mission CSV named a pre-latch commit whose tree REVERSES the scripted
arm's result). A dual-purpose CSV must name the tree that EXECUTED.

Contract enforced here:
  begin(outputs): the working tree must be CLEAN except for the named
      output files (compared by full repo-relative path, so a dirty
      file elsewhere that merely shares a basename does NOT slip
      through); returns the HEAD commit. Raises otherwise — a dirty
      tree produces NO csv, ever.
  finish(token, outputs): HEAD and cleanliness are re-verified at write
      time; the returned commit is the only value a harness may put in
      its commit column. If anything changed mid-run, raise — the run
      is discarded rather than mislabeled.
  verify_csv_commit(csv_path, here): post-hoc self-check for tests and
      re-reviewers — a shipped CSV's git_commit column must hold
      exactly one full hash, that commit must exist in this repo, and
      it must be an ancestor of (or equal to) current HEAD. This does
      NOT prove execution by itself; the proof is begin/finish at
      runtime plus checkout-and-reproduce at the recorded commit.

Scope note: this certifies the REPO tree that executed. Interpreters
and site-packages (numpy etc.) are environment, not repo provenance.

Scope limits (red-team round 2, N1b — read before citing this guard):
  * TRACKED TREE ONLY. The check is `git status --porcelain`, so
    git-IGNORED files are invisible to it: an ignored file that the
    executed code reads or imports is out of scope and its creation
    or modification mid-run is NOT detected. (In this repo the
    ignored artifacts under the harness tree are `__pycache__/*.pyc`
    and an untracked kernel .npz that neither nav-health harness
    loads — hypothetical here, but the limit is structural.)
  * TWO-INSTANT CHECK. Tree state is sampled at begin() and at
    finish() (write time) only. A mid-run excursion that is fully
    undone between the two instants — e.g. commit-then-reset back to
    the token commit, or an edit hidden by
    `git update-index --assume-unchanged` / skip-worktree — is
    undetectable by ANY git-status-based guard and is out of scope.
  What the guard DOES certify, precisely: at the begin instant and
  again at the write instant, HEAD equals the returned commit and
  the non-ignored working tree (tracked modifications, staged
  changes, renames, deletions, and untracked files) contains no
  path outside the named outputs. "Verified executed commit"
  everywhere in this stack means exactly that: verified at begin
  and at write time, under the two limits above.
"""
import csv
import os
import subprocess


def _git(here, *args):
    out = subprocess.run(["git", "-C", here, *args],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr}")
    return out.stdout.strip()


def _dirty_paths(here, outputs):
    allow = {os.path.normpath(os.path.abspath(o)) for o in outputs}
    top = _git(here, "rev-parse", "--show-toplevel")
    status = subprocess.run(
        ["git", "-C", here, "-c", "status.relativePaths=false",
         "status", "--porcelain"],
        capture_output=True, text=True, timeout=15)
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr}")
    dirty = []
    for line in status.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            # Rename/copy (red-team round 2, N1a): BOTH endpoints are
            # checked. The source is a tracked file that DISAPPEARS
            # from its recorded path — checking only the destination
            # let `git mv code.py out.csv` slip through, because the
            # destination was an allowed output while the source's
            # disappearance went unchecked. A rename is allowed only
            # if its source is itself an allowed output (and likewise
            # its destination).
            src, dst = (p.strip().strip('"')
                        for p in path.split(" -> ", 1))
            for p in (src, dst):
                abspath = os.path.normpath(os.path.join(top, p))
                if abspath not in allow:
                    dirty.append(p)
            continue
        abspath = os.path.normpath(os.path.join(top, path))
        if abspath not in allow:
            dirty.append(path)
    return dirty


def begin(here, outputs):
    """Verify a clean tree (outputs excepted); return HEAD commit."""
    dirty = _dirty_paths(here, outputs)
    if dirty:
        raise SystemExit(
            f"PROVENANCE ABORT: working tree dirty ({dirty[:5]}); "
            f"commit before running — a CSV may only name the tree "
            f"that executed it.")
    return _git(here, "rev-parse", "HEAD")


def finish(here, token, outputs):
    """Re-verify at write time; return the executed commit."""
    now = _git(here, "rev-parse", "HEAD")
    if now != token:
        raise SystemExit(
            f"PROVENANCE ABORT: HEAD moved mid-run ({token[:8]} -> "
            f"{now[:8]}); discarding run rather than mislabeling it.")
    dirty = _dirty_paths(here, outputs)
    if dirty:
        raise SystemExit(
            f"PROVENANCE ABORT: tree dirtied mid-run ({dirty[:5]}); "
            f"discarding run rather than mislabeling it.")
    return now


def verify_csv_commit(csv_path, here):
    """Assert a shipped CSV names one real, reachable executed commit.

    Returns the recorded commit. Raises AssertionError (loudly) on:
    no git_commit column, zero/multiple distinct values, a value that
    is not a full 40-hex hash, a commit missing from this repo, or a
    commit that is not an ancestor of HEAD (i.e. not the lineage the
    CSV claims to belong to).
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{csv_path}: no data rows"
    assert "git_commit" in rows[0], f"{csv_path}: no git_commit column"
    vals = {r["git_commit"].strip() for r in rows}
    assert len(vals) == 1, \
        f"{csv_path}: {len(vals)} distinct git_commit values {vals}"
    commit = vals.pop()
    assert len(commit) == 40 and all(
        c in "0123456789abcdef" for c in commit), \
        f"{csv_path}: git_commit {commit!r} is not a full hash"
    kind = subprocess.run(["git", "-C", here, "cat-file", "-t", commit],
                          capture_output=True, text=True, timeout=15)
    assert kind.returncode == 0 and kind.stdout.strip() == "commit", \
        f"{csv_path}: recorded commit {commit} not in this repo"
    anc = subprocess.run(
        ["git", "-C", here, "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True, text=True, timeout=15)
    assert anc.returncode == 0, \
        f"{csv_path}: recorded commit {commit} is not an ancestor of HEAD"
    return commit
