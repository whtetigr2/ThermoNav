"""Tests for the executed-tree provenance guard (round-1 F1 system fix).

Behavioral, per lab doctrine: the guard must be SEEN to bite.
Each test builds a throwaway git repo in a temp dir (never touches the
real tree):
1. Clean tree -> begin returns HEAD; finish returns the same commit.
2. Dirty tree at begin -> SystemExit (a dirty tree produces NO csv).
3. Tree dirtied mid-run -> finish raises SystemExit.
4. HEAD moved mid-run (a commit lands during the run — the exact
   round-1 mission-CSV failure) -> finish raises SystemExit.
5. The named OUTPUT file itself being modified is the one allowed
   exception, and a dirty file in ANOTHER directory that merely
   shares the output's basename is NOT allowed (full-path compare).
6. The shipped nav-health CSVs pass verify_csv_commit: one full hash
   per file, present in this repo, ancestor of HEAD.
7. Round-2 N1a regression: renaming a TRACKED file onto the allowed
   output path (git mv code.py out.csv) aborts at begin and at finish
   — the rename's source disappearance is checked, not just its
   destination. Renaming an allowed output to another allowed output
   remains allowed (control).
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exec_provenance as ep  # noqa: E402


def _git(d, *args):
    subprocess.run(["git", "-C", d, *args], check=True,
                   capture_output=True, text=True)


def _repo():
    d = tempfile.mkdtemp(prefix="provtest_")
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    with open(os.path.join(d, "code.py"), "w") as f:
        f.write("X = 1\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "one")
    return d


def _head(d):
    return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def _expect_abort(fn, label):
    try:
        fn()
    except SystemExit as e:
        assert "PROVENANCE ABORT" in str(e), (label, e)
        print(f"{label}: aborted loudly ({str(e)[:60]}...)")
        return
    raise AssertionError(f"{label}: guard did NOT abort")


def test_clean_roundtrip():
    d = _repo()
    out = os.path.join(d, "out.csv")
    tok = ep.begin(d, [out])
    assert tok == _head(d)
    assert ep.finish(d, tok, [out]) == tok
    print("clean tree: begin/finish return HEAD")


def test_dirty_at_begin_aborts():
    d = _repo()
    with open(os.path.join(d, "code.py"), "a") as f:
        f.write("X = 2\n")
    _expect_abort(lambda: ep.begin(d, [os.path.join(d, "out.csv")]),
                  "dirty at begin")


def test_dirtied_mid_run_aborts():
    d = _repo()
    out = os.path.join(d, "out.csv")
    tok = ep.begin(d, [out])
    with open(os.path.join(d, "other.py"), "w") as f:
        f.write("Y = 1\n")
    _expect_abort(lambda: ep.finish(d, tok, [out]), "dirtied mid-run")


def test_head_moved_mid_run_aborts():
    d = _repo()
    out = os.path.join(d, "out.csv")
    tok = ep.begin(d, [out])
    with open(os.path.join(d, "code.py"), "a") as f:
        f.write("X = 3\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "two")
    _expect_abort(lambda: ep.finish(d, tok, [out]), "HEAD moved mid-run")


def test_output_exception_is_full_path():
    d = _repo()
    out = os.path.join(d, "out.csv")
    tok = ep.begin(d, [out])
    with open(out, "w") as f:  # the harness writing its own output
        f.write("a,b\n1,2\n")
    assert ep.finish(d, tok, [out]) == tok
    # Same basename, different directory: NOT the output -> abort.
    os.mkdir(os.path.join(d, "sub"))
    with open(os.path.join(d, "sub", "out.csv"), "w") as f:
        f.write("impostor\n")
    _expect_abort(lambda: ep.finish(d, tok, [out]),
                  "same-basename impostor")


def test_rename_onto_output_aborts():
    # Round-2 red-team attack A8 / caveat N1a: `git mv code.py out.csv`
    # mid-run — the rename's DESTINATION is the allowed output, but the
    # SOURCE (a tracked file) disappears from its recorded path. The
    # guard must abort at finish, exactly like any other dirty state.
    d = _repo()
    out = os.path.join(d, "out.csv")
    tok = ep.begin(d, [out])
    _git(d, "mv", "code.py", "out.csv")
    _expect_abort(lambda: ep.finish(d, tok, [out]),
                  "rename tracked file onto output mid-run")
    # Same attack staged before begin must also abort at begin.
    d2 = _repo()
    _git(d2, "mv", "code.py", "out.csv")
    _expect_abort(lambda: ep.begin(d2, [os.path.join(d2, "out.csv")]),
                  "rename tracked file onto output at begin")
    # Control: renaming the OUTPUT itself (source IS an allowed
    # output) onto another allowed output stays allowed.
    d3 = _repo()
    out3 = os.path.join(d3, "out.csv")
    out3b = os.path.join(d3, "out2.csv")
    with open(out3, "w") as f:
        f.write("a,b\n")
    _git(d3, "add", "out.csv")
    _git(d3, "commit", "-qm", "out")
    tok3 = ep.begin(d3, [out3, out3b])
    _git(d3, "mv", "out.csv", "out2.csv")
    assert ep.finish(d3, tok3, [out3, out3b]) == tok3
    print("rename of an allowed output to an allowed output: allowed")


def test_shipped_csvs_verify():
    for name in ("nav_health_calibration.csv",
                 "nav_health_mission_validation.csv"):
        commit = ep.verify_csv_commit(os.path.join(HERE, name), HERE)
        print(f"shipped {name}: recorded commit {commit[:8]} exists "
              f"and is an ancestor of HEAD")
        with open(os.path.join(HERE, name), newline="") as f:
            rows = list(csv.DictReader(f))
        assert all(r["git_commit"] == commit for r in rows)


if __name__ == "__main__":
    test_clean_roundtrip()
    test_dirty_at_begin_aborts()
    test_dirtied_mid_run_aborts()
    test_head_moved_mid_run_aborts()
    test_output_exception_is_full_path()
    test_rename_onto_output_aborts()
    test_shipped_csvs_verify()
    print("ALL EXEC-PROVENANCE TESTS PASS")
