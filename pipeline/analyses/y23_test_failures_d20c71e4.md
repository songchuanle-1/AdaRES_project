# Y23 Test Helper Failures

- Local bundle audit first attempt failed before SSH/sync:
  UnicodeDecodeError, Windows default GBK cannot decode inherited UTF-8 source.
- Repair: explicit UTF-8 source read. Original production W2/Y18 unchanged.
- This is a local helper failure, not a training crash or scientific PASS.
