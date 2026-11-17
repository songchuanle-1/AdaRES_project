"""Build a reproducible, non-mutating AdaRES experiment handoff snapshot."""

import argparse
import ast
import base64
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/handoff_20261117"
ARCHIVE = ROOT / "archives/handoff_20261117"
SNAPSHOT = ARCHIVE / "snapshot"
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
GOAL_SHA = "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a"
CODE_EXTS = {".py", ".sh", ".ps1", ".bat", ".cmd", ".cpp", ".cc", ".c", ".cu",
             ".h", ".hpp", ".cuh", ".cmake", ".toml", ".yaml", ".yml"}
BINARY_EXTS = {".pth", ".pt", ".pkl", ".pickle", ".npy", ".npz", ".bin", ".index",
               ".so", ".dll", ".pyd", ".zip", ".tar", ".gz", ".egg", ".whl"}
SECRET_RULES = [
    ("private_key", re.compile(rb"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("github_pat", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("openai_key", re.compile(rb"\bsk-[A-Za-z0-9_-]{35,}\b")),
    ("aws_access_key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("url_user_password", re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@")),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, binary=False):
    output = subprocess.check_output(["git", "-C", str(ROOT)] + list(args))
    return output if binary else output.decode("utf-8").strip()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def assert_invariants():
    if sha((ROOT / "pipeline/goal.json").read_bytes()) != GOAL_SHA:
        raise RuntimeError("Scientific goal changed")
    owner = load(ROOT / "pipeline/execution_mode.json")["target_thread_id"]
    if os.environ.get("CODEX_THREAD_ID") != owner:
        raise RuntimeError("Not the verified owning conversation")
    if (ROOT / "pipeline/headless/.foreground.lockdir").exists():
        raise RuntimeError("Foreground writer exists; do not snapshot concurrently")


def parse_ledger(text):
    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    legacy = ("finished_at", "id", "config", "server", "gpu", "state",
              "best_miou", "epochs", "log_local")
    rows, errors = [], []
    for number, values in enumerate(reader, 2):
        if not values:
            continue
        if len(values) == len(legacy) and re.fullmatch(r"\d{4}-\d{2}-\d{2} .*", values[0]):
            row = dict(zip(legacy, values))
        elif len(values) == len(header):
            row = dict(zip(header, values))
        else:
            errors.append({"line": number, "columns": len(values)})
            continue
        if not re.fullmatch(r"AdaRES-[A-Z]\d+(?:-[A-Z0-9]+)?", row.get("id", "")):
            errors.append({"line": number, "invalid_id": row.get("id")})
            continue
        row["source_line"] = number
        rows.append(row)
    return rows, errors


def number(value):
    try:
        return float(value) if value not in ("", None) else None
    except (ValueError, TypeError):
        return None


def collect_experiments():
    ledger, errors = parse_ledger((ROOT / "pipeline/experiments.csv").read_text("utf-8-sig"))
    by_id = defaultdict(list)
    for row in ledger:
        by_id[row["id"]].append(row)
    registries = {}
    for path in sorted((ROOT / "pipeline/experiments").glob("*.json")):
        record = load(path)
        if record.get("id"):
            registries[record["id"]] = (record, path.relative_to(ROOT).as_posix())
    candidates = load(ROOT / "pipeline/research_work.json")["candidates"]
    candidate_map = {row["id"]: row for row in candidates}
    if len(candidate_map) != len(candidates):
        raise RuntimeError("Duplicate stable candidate IDs")
    reviews = {}
    for path in sorted((ROOT / "pipeline/analyses").glob("*/complete_scientific_review.json")):
        review = load(path)
        if review.get("id"):
            reviews[review["id"]] = (review, path.relative_to(ROOT).as_posix())
    admissions = {}
    for path in sorted((ROOT / "pipeline/admissions").glob("*.json")):
        value = load(path)
        if value.get("id"):
            admissions[value["id"]] = value
    records = []
    for exp_id in sorted(set(by_id) | set(registries) | set(candidate_map), key=id_sort):
        registry, registry_path = registries.get(exp_id, ({}, ""))
        history_rows = by_id.get(exp_id, [])
        legacy = history_rows[-1] if history_rows else {}
        candidate = candidate_map.get(exp_id, {})
        review, review_path = reviews.get(exp_id, ({}, ""))
        terminal = registry.get("terminal_checkpoint_metadata", [])
        final = registry.get("final", {})
        metric_options = [(row.get("best_miou"), "mixed_schema_ledger")
                          for row in history_rows]
        metric_options += [(final.get("best_miou"), "registry.final"),
                           (registry.get("final_best"), "registry.final_best")]
        metric_options += [(row.get("best_miou"), "verified_terminal_checkpoint")
                           for row in terminal]
        metric, metric_source = None, ""
        for value, source in metric_options:
            parsed = number(value)
            if parsed is not None:
                metric, metric_source = parsed, source
        config = (registry.get("config") or legacy.get("config") or candidate.get("config")
                  or admissions.get(exp_id, {}).get("config", ""))
        checkpoints = [row.get("path") for row in terminal if row.get("path")]
        if final.get("checkpoint"):
            checkpoints.append(final["checkpoint"])
        if registry.get("run", {}).get("work_dir"):
            checkpoints.append(registry["run"]["work_dir"])
        for key in ("resume_checkpoint", "resumed_work_dir"):
            if registry.get(key):
                checkpoints.append(registry[key])
        launch_recorded = bool(
            registry.get("history") or registry.get("run") or history_rows
            or registry.get("status") in {"running", "paused", "done_complete", "done_earlystop", "closed_low_metric"}
        )
        records.append({
            "id": exp_id,
            "training_recorded": launch_recorded,
            "training_status": registry.get("status") or legacy.get("status") or legacy.get("state", ""),
            "candidate_stage": candidate.get("stage", ""),
            "config": config,
            "model_package": Path(config).stem.removesuffix("_res") if config else "",
            "hypothesis": candidate.get("hypothesis") or registry.get("design", {}).get("purpose") or registry.get("label", ""),
            "server": registry.get("server") or legacy.get("server", ""),
            "gpu": str(registry.get("gpu", registry.get("gpus", legacy.get("gpu", legacy.get("gpus", ""))))),
            "epochs_recorded": str(registry.get("epochs_final", registry.get("epochs", legacy.get("epochs", "")))),
            "training_best_miou": metric,
            "training_metric_source": metric_source,
            "formal_FIRST_miou": review.get("FIRST_miou"),
            "formal_disposition": review.get("disposition", ""),
            "formal_review_path": review_path,
            "formal_review_sha256": sha((ROOT / review_path).read_bytes()) if review_path else "",
            "failure_scope": candidate.get("failure_scope", review.get("failure_scope", "")),
            "code_commit": registry.get("code_commit") or candidate.get("code_commit", ""),
            "registry_path": registry_path,
            "ledger_lines": [row["source_line"] for row in history_rows],
            "remote_log": registry.get("log") or registry.get("run", {}).get("remote_log", ""),
            "remote_checkpoint_paths": sorted(set(checkpoints)),
            "next_action": candidate.get("next_action", ""),
        })
    trained = [row for row in records if row["training_recorded"]]
    summary = {
        "recorded_experiment_ids": len(records),
        "training_recorded_unique_ids": len(trained),
        "training_status_counts": dict(sorted(Counter(row["training_status"] for row in trained).items())),
        "candidate_records": len(candidates),
        "candidate_stage_counts": dict(sorted(Counter(row["stage"] for row in candidates).items())),
        "ledger_rows": len(ledger),
        "ledger_unique_ids": len(by_id),
        "ledger_parse_errors": errors,
        "registry_records": len(registries),
        "formal_complete_scientific_review_files": len(reviews),
        "formal_disposition_counts": dict(sorted(Counter(row[0].get("disposition", "") for row in reviews.values()).items())),
        "admission_receipt_files": len(list((ROOT / "pipeline/admissions").glob("*.json"))),
        "counts_are_not_success_rates": True,
        "terminal_status_does_not_imply_checkpoint_integrity_or_scientific_acceptance": True,
    }
    return summary, records


def id_sort(exp_id):
    match = re.match(r"AdaRES-([A-Z])(\d+)(.*)", exp_id)
    return (match[1], int(match[2]), match[3]) if match else (exp_id, 0, "")


def export_statistics():
    assert_invariants()
    summary, records = collect_experiments()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(DOCS / "experiment_inventory.json",
               {"generated_at": stamp, "source_head": git("rev-parse", "HEAD"),
                "goal_sha256": GOAL_SHA, "summary": summary, "experiments": records})
    with (DOCS / "experiment_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})
    lines = [
        "# AdaRES 实验统计",
        "",
        f"统计时间（UTC）：{stamp}。原始源提交：`{git('rev-parse', 'HEAD')}`。",
        "",
        "## 统计口径",
        "",
        "按稳定实验 ID 去重，合并混合格式 CSV、逐实验注册表和候选台账。",
        "候选不等于实际训练；延长训练的 B2/B4-E30 等保留原 ID，不能据此计算独立从头训练次数。",
        "注册表终态不自动代表 checkpoint 完整性；训练 best 不等于冻结 FIRST，也不等于已证实改进。",
        "",
        f"- 有训练记录的唯一实验 ID：**{summary['training_recorded_unique_ids']}**。",
        f"- 训练状态：`{json.dumps(summary['training_status_counts'], ensure_ascii=False)}`。",
        f"- 候选记录：**{summary['candidate_records']}**，阶段：`{json.dumps(summary['candidate_stage_counts'])}`。",
        f"- 合并后涉及的唯一 ID：**{summary['recorded_experiment_ids']}**。",
        f"- 原始 CSV：{summary['ledger_rows']} 行，{summary['ledger_unique_ids']} 个 ID；注册表 {summary['registry_records']} 份。",
        f"- 新式 `complete_scientific_review.json`：{summary['formal_complete_scientific_review_files']} 份；旧式完整后评需查原始分析，不能把此数当作全部后评数。",
        f"- 准入收据文件：{summary['admission_receipt_files']}，可能包含修订，不是训练次数。",
        "",
        "## 有训练记录的实验",
        "",
        "| 实验 | 训练状态 | 训练 best mIoU | 新式正式 FIRST | 正式结论/当前候选阶段 |",
        "|---|---|---:|---:|---|",
    ]
    for row in records:
        if not row["training_recorded"]:
            continue
        train = "" if row["training_best_miou"] is None else f"{row['training_best_miou']:.9f}"
        formal = "" if row["formal_FIRST_miou"] is None else f"{row['formal_FIRST_miou']:.9f}"
        lines.append(f"| {row['id']} | {row['training_status']} | {train} | {formal} | "
                     f"{row['formal_disposition'] or row['candidate_stage']} |")
    lines += [
        "", "## 仅候选记录", "",
        "下列记录没有在注册表或原始 CSV 中找到训练记录，不能记为完成训练。完整假设、证据路径与下一步见 JSON/CSV。",
        "", "| 实验 | 候选阶段 |", "|---|---|",
    ]
    lines += [f"| {row['id']} | {row['candidate_stage']} |" for row in records if not row["training_recorded"]]
    lines += [
        "", "## 重要解读", "",
        "- Y36 原始完整后评的 FIRST：Y36 0.510716624292、W2 0.510009849923、B1 0.502112417447；来源为 AdaRES-Y36_frozen_review_1fb792ef.md。",
        "- 后续冻结后评各自包含同设备 FIRST 对照，数值可能变化；不能把不同设备或重复次数的基线拼成一套配对结果。",
        "- Y100 训练 best 0.5164485573768616 超过目标 0.515，但冻结正式后评尚未完成，不能宣布科学目标达成。",
        "- Y90/Y92/Y94 完整后评为实质子组伤害，拒绝范围是该实现；Y96/Y99/Y101 为 complete_inconclusive，不晋升。",
        "- Y88 八次原生预测已保存；正式完整比较/成本/后审尚未完成，成本从未启动，server-1 暂停约束仍有效。",
        "- Y102 原进程存在但原生日志停滞；Y57 用户单独暂停；Y103 仅代码准备，未做 native/准入。",
        "- 本表不计算科学成功率；候选 complete、训练 done_complete 与科学结论是三个维度。",
        "",
    ]
    (DOCS / "EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def capture_remote():
    assert_invariants()
    script = (Path(__file__).with_name("collect_remote.py")).read_bytes()
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
         "ClearAllForwardings=yes", "server-2",
         "PYTHONDONTWRITEBYTECODE=1 /home/songcl/.conda/envs/adares_project/bin/python -B - --sources"],
        input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180,
    )
    body = json.loads(result.stdout)
    payload = base64.b64decode(body.pop("source_archive_base64"), validate=True)
    if sha(payload) != body["source_archive_sha256"]:
        raise RuntimeError("Source archive digest mismatch")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE / "server2_source.tar.gz"
    archive_path.write_bytes(payload)
    source_root = ARCHIVE / "server2_source"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError("Unsafe remote archive member")
            data = archive.extractfile(member).read()
            target = source_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    write_json(DOCS / "server2_readonly_inventory.json", body)
    package_lines = [f"{row['name']}=={row['version']}" for row in body["packages"] if row["name"]]
    (DOCS / "runtime_packages_observed.txt").write_text("\n".join(package_lines) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(body["sources"]),
                      "source_archive_bytes": len(payload),
                      "observed_at": body["observed_at"], "y102": body["y102"]}, ensure_ascii=False))


def publication_decision(relative, path):
    parts = PurePosixPath(relative).parts
    if not parts:
        return False, "empty"
    if parts[0] in {"archives", "Checkpoints"}:
        return False, "separate_artifact_transfer"
    if relative in {".pipeline.lock", ".pipeline.lock.owner.json"} or any(
        part in {"__pycache__", ".git", ".pytest_cache"} for part in parts
    ):
        return False, "ephemeral_or_git_internal"
    if relative.startswith("pipeline/headless/") and Path(relative).name.startswith("."):
        return False, "controller_ephemeral_state"
    if re.fullmatch(r"pipeline/analyses/y56_cached_train_.*_program\.py", relative):
        return False, "self_contained_diagnostic_with_embedded_large_training_cache"
    if path.suffix.lower() in BINARY_EXTS:
        return False, "binary_or_training_artifact"
    if path.suffix.lower() in CODE_EXTS:
        return True, "source"
    if relative == "pipeline/research_work.json":
        return True, "complete_candidate_ledger"
    if path.stat().st_size > 1024 * 1024:
        return False, "large_raw_evidence_over_1MiB"
    return True, "small_text_evidence_or_metadata"


def check_secrets(data, name):
    for rule, regex in SECRET_RULES:
        if regex.search(data):
            raise RuntimeError(f"Potential secret ({rule}) in {name}; do not publish")
    if PurePosixPath(name).name.lower() in {".env", ".netrc", ".git-credentials", "id_rsa", "id_ed25519"}:
        raise RuntimeError(f"Credential filename in {name}; do not publish")


def source_files():
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", binary=True)
    return sorted(set(item.decode("utf-8") for item in paths.split(b"\0") if item))


def original_refs():
    rows = git("for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/tags")
    return [{"ref": row.split(" ", 1)[0], "commit": row.split(" ", 1)[1]}
            for row in rows.splitlines()]


def scan_history():
    assert_invariants()
    rows = git("rev-list", "--objects", "--all").splitlines()
    names = {row.split(" ", 1)[0]: row.split(" ", 1)[1] if " " in row else ""
             for row in rows}
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input=("\n".join(names) + "\n").encode(), stdout=subprocess.PIPE, check=True,
    )
    blobs = []
    for row in result.stdout.decode().splitlines():
        oid, kind, size = row.split()
        if kind != "blob":
            continue
        data = git("cat-file", "blob", oid, binary=True)
        check_secrets(data, names[oid])
        if len(data) > 50 * 1024 * 1024:
            raise RuntimeError("Oversize historical blob; do not push")
        blobs.append({"oid": oid, "bytes": int(size), "representative_path": names[oid]})
    report = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "original_refs": original_refs(), "blobs_scanned": len(blobs),
              "blob_bytes": sum(row["bytes"] for row in blobs),
              "largest_blob": max(row["bytes"] for row in blobs),
              "secret_scan_passed": True, "blobs": blobs,
              "limitation": "Pattern-based scan is not a proof of absence of all possible secrets."}
    write_json(DOCS / "history_scan.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"blobs", "original_refs"}}, ensure_ascii=False))


def build_snapshot():
    assert_invariants()
    if SNAPSHOT.exists():
        raise RuntimeError("Snapshot exists; do not overwrite a publication")
    status_before = git("status", "--porcelain=v1", "--untracked-files=no")
    index_path = ROOT / ".git/index"
    index_before = sha(index_path.read_bytes())
    head = git("rev-parse", "HEAD")
    manifest, excluded, conflicts = [], [], []
    SNAPSHOT.mkdir(parents=True)

    def copy(name, data, origin):
        check_secrets(data, name)
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"path": name, "bytes": len(data), "sha256": sha(data), "origin": origin})

    for relative in source_files():
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f"Review local symlink before publication: {relative}")
        include, reason = publication_decision(relative, path)
        data = path.read_bytes()
        if not include:
            excluded.append({"path": relative, "bytes": len(data), "sha256": sha(data), "reason": reason})
            continue
        copy(relative, data, "local_working_tree_exact_bytes")
    branch_only = []
    for ref in original_refs():
        entries = git("ls-tree", "-rz", ref["commit"], binary=True).split(b"\0")
        for entry in entries:
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            relative = raw_name.decode("utf-8")
            path = PurePosixPath(relative)
            if path.parts[0] not in {"AdaRES", "scripts"} or path.suffix not in CODE_EXTS:
                continue
            if (ROOT / relative).exists() or (SNAPSHOT / relative).exists():
                continue
            oid = metadata.decode().split()[-1]
            data = git("cat-file", "blob", oid, binary=True)
            copy(relative, data, "branch_only_source:" + ref["commit"])
            branch_only.append({"path": relative, "commit": ref["commit"], "ref": ref["ref"]})
    remote = ARCHIVE / "server2_source"
    for path in sorted(remote.rglob("*")):
        if not path.is_file():
            continue
        relative = "AdaRES/" + path.relative_to(remote).as_posix()
        data = path.read_bytes()
        local = SNAPSHOT / relative
        if local.exists():
            if local.read_bytes() != data:
                conflicts.append({"path": relative, "local_sha256": sha(local.read_bytes()),
                                  "server2_sha256": sha(data),
                                  "resolution": "local preserved; remote exact bytes under vendor source snapshot"})
                copy("vendor/server2_source/" + path.relative_to(remote).as_posix(), data,
                     "original_server2_readonly_conflict_copy")
            continue
        copy(relative, data, "original_server2_missing_runtime_source")
    for path in sorted(SKILL.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.relative_to(SKILL).parts):
            continue
        if path.suffix.lower() not in CODE_EXTS | {".md", ".txt"}:
            continue
        if path.name in {"smtp_config.json"}:
            continue
        relative = "vendor/songcl_research_pipeline/" + path.relative_to(SKILL).as_posix()
        copy(relative, path.read_bytes(), "canonical_controller_tooling_observational_copy_not_enabled")
    readme = (DOCS / "README.md").read_text("utf-8")
    readme = re.sub(r"\]\(([^:/)]+)\)",
                    lambda match: "](docs/handoff_20261117/" + match[1] + ")", readme)
    copy("README.md", readme.encode("utf-8"), "handoff_navigation")
    ignore = b"""__pycache__/
*.py[cod]
.env
.env.*
.pipeline.lock*
Checkpoints/
data/
AdaRES/data/
AdaRES/backbones/
AdaRES/logs/
AdaRES_checkpoints/
runtime/
*.pth
*.pt
*.pkl
*.pickle
*.npy
*.npz
*.so
*.pyd
*.dll
*.egg
*.whl
"""
    copy(".gitignore", ignore, "snapshot_only_artifact_guard")
    # The source repo's index, HEAD and all scientific files remain untouched.
    if index_before != sha(index_path.read_bytes()) or head != git("rev-parse", "HEAD"):
        raise RuntimeError("Active source HEAD/index changed")
    if status_before != git("status", "--porcelain=v1", "--untracked-files=no"):
        raise RuntimeError("Existing tracked working changes changed")
    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_head": head, "source_index_sha256": index_before,
        "scientific_goal_sha256": GOAL_SHA,
        "files": sorted({row["path"]: row for row in manifest}.values(), key=lambda row: row["path"]),
        "included_file_count": len({row["path"] for row in manifest}),
        "excluded_files": excluded,
        "remote_local_conflicts": conflicts,
        "branch_only_sources_added": branch_only,
        "original_refs": original_refs(),
        "source_HEAD_index_and_existing_tracked_changes_preserved": True,
        "remote_processes_and_scheduler_unchanged": True,
        "secret_scan_rules": [row[0] for row in SECRET_RULES],
    }
    write_json(DOCS / "publication_manifest.json", report)
    write_json(SNAPSHOT / "docs/handoff_20261117/publication_manifest.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in
                      {"files", "excluded_files", "remote_local_conflicts", "branch_only_sources_added", "original_refs"}},
                     ensure_ascii=False))


def verify_snapshot():
    manifest = load(SNAPSHOT / "docs/handoff_20261117/publication_manifest.json")
    failures = []
    checked = {}
    for row in manifest["files"]:
        checked[row["path"]] = row
    for relative, row in checked.items():
        path = SNAPSHOT / relative
        if sha(path.read_bytes()) != row["sha256"]:
            failures.append(relative + ": hash mismatch")
        check_secrets(path.read_bytes(), relative)
        if path.suffix == ".py":
            try:
                ast.parse(path.read_bytes(), filename=relative)
            except (SyntaxError, ValueError) as exc:
                failures.append(relative + ": " + str(exc))
    required = ["AdaRES/ipdn/dataset/__init__.py", "AdaRES/ipdn/utils/__init__.py",
                "AdaRES/ipdn/torch/nn_new/__init__.py", "AdaRES/ipdn/lib/pointgroup_ops/pointgroup_ops.py",
                "AdaRES/tools/train.py", "AdaRES/setup.py", "AdaRES/requirements.txt"]
    missing = [path for path in required if not (SNAPSHOT / path).exists()]
    result = {"manifest_files_checked": len(checked), "parse_or_hash_failures": failures,
              "missing_required_runtime_sources": missing,
              "source_head_unchanged": git("rev-parse", "HEAD") == manifest["source_head"],
              "source_index_unchanged": sha((ROOT / ".git/index").read_bytes()) == manifest["source_index_sha256"],
              "goal_unchanged": sha((ROOT / "pipeline/goal.json").read_bytes()) == GOAL_SHA}
    write_json(DOCS / "snapshot_verification.json", result)
    write_json(SNAPSHOT / "docs/handoff_20261117/snapshot_verification.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if failures or missing or not all(result[key] for key in
               ("source_head_unchanged", "source_index_unchanged", "goal_unchanged")) else 0


def refresh_owned_snapshot():
    assert_invariants()
    path = SNAPSHOT / "docs/handoff_20261117/publication_manifest.json"
    manifest = load(path)
    rows = {row["path"]: row for row in manifest["files"]}
    removed = []
    for relative, row in list(rows.items()):
        original = ROOT / relative
        if not original.is_file():
            continue
        allowed, reason = publication_decision(relative, original)
        if allowed or reason != "self_contained_diagnostic_with_embedded_large_training_cache":
            continue
        target = (SNAPSHOT / relative).resolve()
        if SNAPSHOT.resolve() not in target.parents:
            raise RuntimeError("Snapshot target escaped workspace")
        if sha(target.read_bytes()) != row["sha256"]:
            raise RuntimeError("Do not remove an altered publication file")
        target.unlink()
        del rows[relative]
        removed.append(relative)
        manifest["excluded_files"].append(
            {"path": relative, "bytes": row["bytes"], "sha256": row["sha256"], "reason": reason})
    for relative in ["scripts/handoff_20261117/export.py",
                     "docs/handoff_20261117/MIGRATION_HANDOFF.md"]:
        data = (ROOT / relative).read_bytes()
        check_secrets(data, relative)
        (SNAPSHOT / relative).write_bytes(data)
        rows[relative] = {"path": relative, "bytes": len(data), "sha256": sha(data),
                          "origin": "local_working_tree_exact_bytes"}
    manifest["files"] = sorted(rows.values(), key=lambda row: row["path"])
    manifest["included_file_count"] = len(rows)
    manifest["prepublication_refresh"] = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedded_cache_artifacts_excluded": removed,
        "original_files_not_modified_or_deleted": True,
    }
    write_json(DOCS / "publication_manifest.json", manifest)
    write_json(path, manifest)
    print(json.dumps({"included_files": len(rows), "excluded_embedded_cache_artifacts": removed},
                     ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["statistics", "capture-remote", "scan-history", "build-snapshot",
                                         "refresh-owned-snapshot", "verify"])
    args = parser.parse_args()
    actions = {"statistics": export_statistics, "capture-remote": capture_remote,
               "scan-history": scan_history, "build-snapshot": build_snapshot, "verify": verify_snapshot}
    actions["refresh-owned-snapshot"] = refresh_owned_snapshot
    return actions[args.action]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
