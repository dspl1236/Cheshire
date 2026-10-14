"""Dry run of apply_hip_patch.py on another AliceVision checkout (CHESHIRE_AV_DIR), e.g. a worktree of an upstream PR or
of develop: every anchor that does not apply is recorded and the run goes on, so one pass lists them all. It writes into
that checkout, never into third_party/aliceVision.  py -3 scripts/dryrun_patch.py <AliceVision checkout>"""
import os
import sys
import traceback
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
os.environ["CHESHIRE_AV_DIR"] = str(tree)
os.environ["CHESHIRE_SKIP_PATCH_EXPORT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_hip_patch as ahp  # noqa: E402

failures = []


class _Sys:
    def __getattr__(self, name):
        return getattr(sys, name)

    @staticmethod
    def exit(msg=None):
        where = traceback.extract_stack(limit=2)[0]
        failures.append((where.lineno, str(msg).splitlines()[0] if msg else ""))


ahp.sys = _Sys()
try:
    ahp.main()
    status = "ran to the end"
except Exception as e:  # a later step that tripped over an earlier miss
    status = f"stopped: {type(e).__name__}: {e}"
    failures.append((traceback.extract_tb(e.__traceback__)[-1].lineno, f"EXCEPTION {type(e).__name__}: {str(e)[:200]}"))
print(f"== {tree.name}: {status}; {len(failures)} anchor failure(s)")
for line, msg in failures:
    print(f"  apply_hip_patch.py:{line}  {msg[:220]}")
