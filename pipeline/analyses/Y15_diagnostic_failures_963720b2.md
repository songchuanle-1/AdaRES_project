# Diagnostic Failure And Recovery - AdaRES-Y15

- Tick963720b211ae49b1a59707610cfbda4a; original failures retained, no checkpoint/native model/config/selection changes.
- First frozen-eval helper omitted existing ipdn/lib from isolated Python3.8 import paths; ModuleNotFoundError:pointgroup_ops occurred before native inference. Log y15_y15_963720b2.log SHA256:ad162476746ce83cf3086ae4fd91e575d71cadb6407c4efe03f6fb526226cda7.
- Concrete repair: use existing deployment import path CODE/ipdn/lib, no foreign environment or package installation. New v2 output/log successfully completed9508expressions/twopasses; never overwrite failed attempt.
- First query-decomposition generator replaced the substring CODE inside PYTHONDONTWRITEBYTECODE; SyntaxError before any imports/inference. Log y15_query_decomposition_963720b2.log SHA256:cf3382cbcd2ce392797d2f080aaac5241a46611b504bef2ac95e29141f4f2b0c.
- Concrete repair: distinct __ROOT__/__PATHS__/__CHECKPOINT__/__MANIFEST__ placeholders plus local compile(program) before SSH. New v2 completed9508expressions and native IoU identity checks.
- Read-only post-diagnostic check initially used check_output(pgrep), which raises for normal no-match exit1; no scientific files changed. Replaced only in the read-only check with run, verified no matching process and available advisory flock at2026-10-02T06:12:37+08:00.
- One mistaken local compact-view path and report filename returned file-not-found; canonical headless view/summary path subsequently read. No old snapshot substituted, no second begin or preflight.
- Scope: diagnostic transport/import packaging recovery only. These repairs do not establish a training defect or justify retrying the rejected candidate.
