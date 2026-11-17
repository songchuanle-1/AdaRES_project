# Y42 local build failure and scoped repair

The first build_y42_local_bundle.py invocation failed at parent dec.py
read_text() with UnicodeDecodeError: the Windows default GBK decoder could
not read existing UTF-8 source. It had only copied the six unchanged W2 files
into the new Y42 directory; no configuration, remote file, training or
admission was written. This was a build fault, not a scientific rejection.

Repair: use explicit UTF-8 for source reads. Recovery accepts that existing
directory only if it contains exactly six files, each byte-identical to W2.
It does not delete or overwrite unrelated work. The covariance protocol,
frozen sampling, controls and gates are unchanged.
