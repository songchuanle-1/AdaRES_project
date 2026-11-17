# AdaRES-E1 Stability Diagnosis

2026-09-24 11:21 Asia/Shanghai. Read-only B4-E30 checkpoint probe on server-2 GPU1, fixed first 16 ScanRefer validation scenes (441 expressions). No training, optimizer update, production sync, or queue dispatch.

- Definition: IoU of binary masks at the same query index between the penultimate and final decoder layers, restricted to valid superpoints. Ground truth was used only for diagnostic correlation and selected/oracle comparison, never as inference input.
- Mean within-expression query-rank correlation between stability and true mask IoU: 0.6837. This alone is not evidence of improved top-1 selection.
- Of 240 material selection errors (oracle minus selected IoU >0.05), the oracle was more stable than the selected query in only 56. Mean selected stability 0.9562, oracle stability 0.9123.
- The proposed confidence mechanism, that layer stability distinguishes the missed oracle from the chosen query, fails this bounded test. No candidate package/config or control was implemented; native backward, candidate resource measurement, independence and admission gates were not attempted. E1 is rejected, not admitted.
- This is 16 scenes, not a full-validation causal study. Another definition of stability might behave differently, but choosing one by validation would be a feature/threshold scan prohibited by the goal.
- Native probe script: `AdaRES/tools/diagnose_adares_e1_stability.py`; result printed by native `adares_project` environment on GPU1. Two pre-model failures were repaired (staged import path and custom `.cuda()` return); an initial device-helper override attempted occupied GPU0 and OOMed during placement. Final successful run explicitly selected physical GPU1 and left D1 unchanged.

Next: monitor D1; pursue a different independently diagnosed information path only with prospective controls, native backward and resources. Do not enqueue E1 or reinterpret this probe as a candidate evaluation.
