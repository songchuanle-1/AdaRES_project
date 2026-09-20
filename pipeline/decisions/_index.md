# AdaRES_project 决策索引

| 时间 | 类型 | 问题 | 决策 |
|------|------|------|------|
| 2026-09-18 01:15:00 | [route_change](20260918-011500_route_change.md) | B0 finished at mIoU 0.333; is the failure masks or ranking? | Close B0; rebuild with a faithful standard-indicator baseline |
| 2026-09-18 01:20:00 | [route_change](20260918-012000_route_change.md) | How to avoid another idle period after B1? | Deploy a gated watchdog that smoke-tests and launches B2 after B1 |
| 2026-09-20 09:14:44 | [other](20260920-091444_other.md) | Did the failed B2 smoke invalidate the model? | Fix the model-agnostic smoke harness, revalidate, and launch B2 |
| 2026-09-20 09:20:00 | [manual](20260920-092000_manual.md) | Should new experiments require a user message? | Enable hourly autonomous research heartbeat and successor queue replenishment |
