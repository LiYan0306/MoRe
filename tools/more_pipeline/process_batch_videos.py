#!/usr/bin/env python3
"""Batch controller: check ALL -> smoke ALL -> run each sequentially."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "sequence", "video", "status", "frames", "fps",
    "extract_s", "inference_s", "visualize_s", "total_s",
    "extract_skipped", "inference_skipped", "visualize_skipped", "error", "log_dir",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/MoRe"))
    parser.add_argument("--video-dir", type=Path, help="默认 ROOT/data-tiny/raw-video")
    parser.add_argument("--preflight-only", action="store_true", help="仅检查全部视频/环境")
    parser.add_argument("--smoke-only", action="store_true", help="检查+逐视频小样本端到端测试")
    parser.add_argument("--skip-smoke", action="store_true", help="跳过端到端小样本验证（不推荐首次使用）")
    parser.add_argument("--smoke-frames", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="继续上次中断的正式批处理")
    args = parser.parse_args()
    if args.smoke_frames < 1:
        parser.error("--smoke-frames 必须 >= 1")
    if args.preflight_only and args.smoke_only:
        parser.error("--preflight-only 和 --smoke-only 不能同时使用")
    if args.smoke_only and args.skip_smoke:
        parser.error("--smoke-only 和 --skip-smoke 不能同时使用")

    root = args.root.resolve()
    video_dir = (args.video_dir or root / "data-tiny" / "raw-video").resolve()
    if not video_dir.is_dir():
        parser.error(f"视频目录不存在: {video_dir}")
    videos = sorted(p for p in video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        parser.error(f"找不到 MP4 视频: {video_dir}")
    stems = [p.stem for p in videos]
    if len(stems) != len(set(stems)):
        parser.error("存在同名不同大小写扩展名的视频，会导致输出冲突")

    single = Path(__file__).with_name("process_one_video.py")
    if not single.is_file():
        parser.error(f"找不到单视频处理脚本: {single}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    out_dir = root / "results" / "pipeline_logs" / (f"batch_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=False)
    csv_file = out_dir / "batch_summary.csv"
    records = []

    def flush_csv() -> None:
        with csv_file.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({field: record.get(field, "") for field in FIELDS}
                             for record in records)

    def call(video: Path, mode: str) -> tuple[int, dict]:
        summary_path = out_dir / f"{video.stem}_{mode}.json"
        cmd = [sys.executable, str(single), "--root", str(root),
               "--video", str(video), "--mode", mode,
               "--smoke-frames", str(args.smoke_frames),
               "--summary-json", str(summary_path)]
        if mode == "run" and args.resume:
            cmd += ["--resume"]
        print(f"\n[BATCH {mode.upper()}] {video.name}", flush=True)
        try:
            code = subprocess.call(cmd)
        except OSError as exc:
            code = 1
            print(f"[ERROR] 启动失败: {exc}", file=sys.stderr, flush=True)
        if summary_path.is_file():
            try:
                row = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                row = {}
        else:
            row = {}
        row.setdefault("sequence", video.stem)
        row.setdefault("video", str(video))
        if code != 0:
            row["status"] = "FAILED"
            row["error"] = row.get("error") or f"{mode} 进程退出码 {code}"
        return code, row

    # Phase 1: no full GPU job is launched until every input passes static checks.
    for video in videos:
        code, row = call(video, "check")
        if code != 0:
            records.append(row)
            flush_csv()
            print(f"[ABORT] 预检查失败；汇总: {csv_file}", file=sys.stderr)
            return 1
    if args.preflight_only:
        print(f"[OK] {len(videos)} 个视频预检查通过。")
        return 0

    # Phase 2: verify the complete 3-stage path on a short clip from EVERY video.
    if not args.skip_smoke:
        for video in videos:
            code, row = call(video, "smoke")
            if code != 0:
                records.append(row)
                flush_csv()
                print(f"[ABORT] 小样本端到端验证失败；汇总: {csv_file}", file=sys.stderr)
                return 1
    if args.smoke_only:
        print(f"[OK] {len(videos)} 个视频小样本端到端验证通过。")
        return 0

    # Phase 3: actual production; ONE video at a time (bounded GPU memory pressure).
    for i, video in enumerate(videos, 1):
        print(f"\n[PRODUCTION {i}/{len(videos)}] {video.name}", flush=True)
        code, row = call(video, "run")
        records.append(row)
        flush_csv()  # durability: keep successful timings even when a later run fails
        if code != 0:
            print(f"[ABORT] 第 {i} 个视频失败；可排查后加 --resume 重启批次。汇总: {csv_file}",
                  file=sys.stderr, flush=True)
            return 1
    print(f"\n[OK] 全部 {len(videos)} 个序列处理成功；耗时表: {csv_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
