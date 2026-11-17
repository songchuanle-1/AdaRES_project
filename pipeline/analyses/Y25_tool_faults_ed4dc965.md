# Y25 Diagnostic Tool Faults
Token ed4dc9656e8f4556a57c17cb369279c6, 2026-10-03.

- First build_y25_test_bundle.py invocation failed at line32 local
  ast.parse(path.read_text()) with UnicodeDecodeError: GBK cannot decode byte
  0xa6 at position16155. No SSH/bundle transmission/production training occurred;
  failure precedes remote invocation and local monitor lock acquisition.
  Repair: explicitly UTF-8 for local Python source reads, not model alteration.
  Subsequent actual result must be recorded independently; no assumed PASS.
- Y24 initial target-gradient probe failed its raw source SHA assertion.
  First raw local/remote hashes differ; LF normalization did not equate them.
  Cross-Python3.13/3.8 AST hashes also differ and do not prove semantic mismatch.
  Downloaded deployed source parsed beside local source in the SAME interpreter:
  five tested functions and entire module AST equal. Corrected probe binds
  exact deployed source SHA, new v2 output passes32 actual native cases.
  Original first-probe log remains immutable. No original model source edited.
- First y25_resources probe completed native training measurements in memory
  but failed its first inference assertion `pred_pmask.shape[0] == 128`.
  Native RES predict returns only the selected point mask; query confidence
  vector `indi` carries128 query scores. This is a measuring-tool interface
  assertion fault, not native model crash/OOM or evidence of128 points.
  Repair: assert indi128 and selected point-mask length equals actual points.
  Rerun unique y25_resources_v2; no rewrite of original log, no gate loosening.
