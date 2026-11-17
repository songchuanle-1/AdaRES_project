# Y44 Local Upload Reader Repair

Observed2026-10-13 06:12+08 during ownedff347697 tick.
First upload exited1 at local ast.parse(path.read_text()) with GBK
UnicodeDecodeError byte0x9f position2052. This occurred before lock acquisition
and SSH invocation; no remote files written by that failed attempt.
Repair only explicit UTF-8 local source/parent reading; model files,
configuration, source manifest, gates, tolerances and prospective controls
unchanged. Failed tool output retained in current task. Not a scientific test.
