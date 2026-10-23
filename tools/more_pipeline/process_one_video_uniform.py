#!/usr/bin/env python3
"""单视频：按原始帧序号均匀抽取 N 帧 -> MoRe 推理 -> motion mask 可视化。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def ensure(ok: bool, reason: str) -> None:
    if not ok:
        raise PipelineError(reason)


def fingerprint(path: Path) -> dict:
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def probe(video: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
           "-of", "json", str(video)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    ensure(p.returncode == 0, f"ffprobe 失败: {p.stderr.strip()}")
    streams = json.loads(p.stdout).get("streams", [])
    ensure(len(streams) == 1, "没有找到可解码的视频流")
    v = streams[0]
    total = int(v.get("nb_read_frames") or 0)
    ensure(total > 0, "无法可靠统计源视频总帧数")
    ensure(int(v.get("width") or 0) > 0 and int(v.get("height") or 0) > 0,
           "视频分辨率无效")
    fps = v.get("avg_frame_rate", "0/0")
    try:
        if Fraction(fps) <= 0:
            fps = v.get("r_frame_rate", "0/0")
        ensure(Fraction(fps) > 0, "视频帧率无效")
    except (ValueError, ZeroDivisionError) as exc:
        raise PipelineError("视频帧率解析失败") from exc
    return {"total": total, "fps": fps, "width": int(v["width"]),
            "height": int(v["height"])}


def sample_positions(total: int, count: int) -> list[int]:
    """首尾均包含；等距实数位置向上取整。count <= total；没有重复。"""
    ensure(1 <= count <= total, "采样数量无效")
    if count == 1:
        return [(total - 1) // 2]
    d = count - 1
    return [(i * (total - 1) + d - 1) // d for i in range(count)]


def sample_filter(total: int, count: int) -> str | None:
    """只用 O(1) 长度 FFmpeg select 表达式，选择与 sample_positions 完全相同的帧。"""
    if count == total:
        return None
    if count == 1:
        return f"select=eq(n\\,{(total - 1) // 2})"
    a, b = count - 1, total - 1
    return f"select=gt(floor(n*{a}/{b})\\,floor((n-1)*{a}/{b}))"


def checked_files(path: Path, regex: re.Pattern, n: int, label: str,
                  continuous: bool = True) -> None:
    ensure(path.is_dir(), f"{label} 目录不存在: {path}")
    found = sorted(p for p in path.iterdir() if p.is_file() and regex.fullmatch(p.name))
    ensure(len(found) == n, f"{label} 数量不符: 预期 {n}，实际 {len(found)}，目录 {path}")
    ensure(all(p.stat().st_size > 0 for p in found), f"{label} 存在空文件: {path}")
    if continuous:
        indices = sorted(int(regex.fullmatch(p.name).group(1)) for p in found)
        ensure(indices == list(range(n)), f"{label} 文件序号不连续: {path}")


FRAME = re.compile(r"frame_(\d+)\.png", re.I)
MASK = re.compile(r"mask_(\d+)\.png", re.I)
PLY = re.compile(r"frame_(\d+)\.ply", re.I)
NPY = re.compile(r"(\d+)\.npy", re.I)
IMAGE = re.compile(r"[^.].*\.(png|jpg|jpeg)", re.I)


def verify_extract(directory: Path, expected: int) -> None:
    checked_files(directory, FRAME, expected, "抽帧 PNG")
    ensure(all(p.is_file() and FRAME.fullmatch(p.name) for p in directory.iterdir()),
           f"图片目录含非抽帧 PNG 文件，可能影响 MoRe 读取: {directory}")


def verify_inference(directory: Path, n: int) -> None:
    checked_files(directory / "masks", MASK, n, "motion mask")
    checked_files(directory, PLY, n, "逐帧 PLY")
    for name in ("depth", "extrinsic", "intrinsic"):
        checked_files(directory / name, NPY, n, name + " NPY")


def verify_overlay(directory: Path, n: int) -> None:
    checked_files(directory, IMAGE, n, "叠加可视化", continuous=False)


def check_deps(root: Path, video: Path) -> dict:
    for cmd in ("ffmpeg", "ffprobe"):
        ensure(shutil.which(cmd) is not None, f"未找到命令 {cmd}")
    ensure(video.is_file() and video.stat().st_size > 0, f"视频不存在或为空: {video}")
    code = root / "code" / "MoRe"
    files = {
        "inference": code / "inference.py",
        "visualize": code / "test" / "visualize_more_motion_mask.py",
        "config": code / "training" / "config" / "omniworld_full.yaml",
        "weight": code / "pretrained" / "more_full.pt",
        "skyseg": code / "pretrained" / "skyseg.onnx",
    }
    for key, path in files.items():
        ensure(path.is_file() and path.stat().st_size > 0,
               f"缺少 {key} 文件: {path}")
    for key in ("inference", "visualize"):
        compile(files[key].read_bytes(), str(files[key]), "exec")
    return {"code": code, **files}


def run_logged(command: list[str], cwd: Path, log: Path,
               detect_oom: bool = False) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("CMD:", " ".join(map(str, command)), flush=True)
    saw_oom = False
    with log.open("w", encoding="utf-8") as out:
        out.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        out.flush()
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                              text=True, bufsize=1, start_new_session=True) as proc:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    out.write(line)
                    print(line, end="", flush=True)
                    lower = line.lower()
                    if "out of memory" in lower and ("cuda" in lower or "gpu" in lower):
                        saw_oom = True
                code = proc.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    pass
                raise
    if detect_oom and saw_oom:
        raise PipelineError(f"MoRe 推理显存不足（CUDA OOM），请调低 --frames；详见 {log}")
    ensure(code == 0, f"子进程退出码 {code}，详见日志: {log}")


def extract(video: Path, temp: Path, count: int, total: int,
            cwd: Path, log: Path) -> None:
    temp.mkdir(parents=True, exist_ok=False)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
           "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn"]
    select = sample_filter(total, count)
    if select is not None:
        cmd += ["-vf", select]
    cmd += ["-fps_mode", "passthrough", "-start_number", "0",
            "-compression_level", "3", str(temp / "frame_%08d.png")]
    run_logged(cmd, cwd, log)


def infer(images: Path, temp: Path, code: Path, log: Path) -> None:
    ensure(not temp.exists(), f"临时输出已存在: {temp}")
    run_logged([sys.executable, "inference.py", "--config_path",
                "training/config/omniworld_full.yaml", "--ckpt_path", "pretrained/more_full.pt",
                "--image_path", str(images), "--output_dir", str(temp),
                "--skyseg_onnx", "pretrained/skyseg.onnx",
                "--conf_thres", "50.0", "--predict_motion"], code, log, detect_oom=True)


def overlay(images: Path, inference: Path, temp: Path, code: Path, log: Path) -> None:
    ensure(not temp.exists(), f"临时输出已存在: {temp}")
    run_logged([sys.executable, "test/visualize_more_motion_mask.py",
                "--image-dir", str(images), "--mask-dir", str(inference / "masks"),
                "--output-dir", str(temp), "--opacity", "0.85", "--gamma", "1.0",
                "--threshold", "0.0", "--align", "more-crop"], code, log)


@contextmanager
def exclusive_lock(root: Path, sequence: str):
    lock_dir = root / "results" / "pipeline_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / (sequence + ".lock")).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"序列正在被另一个进程处理: {sequence}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process(root: Path, video: Path, requested: int, mode: str, resume: bool) -> int:
    root, video = root.resolve(), video.resolve()
    sequence = f"{video.stem}_n{requested}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    log_dir = root / "results" / "pipeline_logs" / sequence / run_id
    summary = log_dir / "summary.json"
    record = {"video": str(video), "sequence": sequence, "requested_frames": requested,
              "source_frames": None, "sampled_frames": None, "mode": mode,
              "status": "FAILED", "extract_s": None, "inference_s": None,
              "visualize_s": None, "extract_skipped": False,
              "inference_skipped": False, "visualize_skipped": False,
              "error": "", "log_dir": str(log_dir)}
    begun = time.perf_counter()
    try:
        with exclusive_lock(root, sequence):
            deps = check_deps(root, video)
            info = probe(video)
            count = min(requested, info["total"])
            indices = sample_positions(info["total"], count)
            record.update(source_frames=info["total"], sampled_frames=count,
                          fps=info["fps"], resolution=f"{info['width']}x{info['height']}")
            print(f"[CHECK] {video.name}: {info['total']} source frames, "
                  f"{info['width']}x{info['height']}, fps={info['fps']}", flush=True)
            print(f"[PLAN] 目标 {requested} 帧，实际 {count} 帧；"
                  f"采样序号前5={indices[:5]}，后5={indices[-5:]}", flush=True)
            if count < requested:
                print("[WARN] N 大于源帧数：只保留源视频所有帧，不复制补帧", flush=True)
            if mode == "check":
                record["status"] = "OK"
                return 0

            image_dir = root / "data-tiny" / sequence
            inference_dir = root / "results" / f"{sequence}_full_glbfix"
            overlay_dir = root / "results" / f"{sequence}_motion_overlay"
            state_file = root / "results" / "pipeline_state" / (sequence + ".json")
            state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
            manifest = {"method": "uniform_frame_index_ceil_v1", "video": fingerprint(video),
                        "source": info, "requested_frames": requested,
                        "sampled_frames": count, "source_frame_indices_zero_based": indices}
            write_json(log_dir / "sampling.json", manifest)
            record["sampling_map"] = str(log_dir / "sampling.json")
            extract_sig = {"manifest": manifest, "format": "png", "filter": sample_filter(info["total"], count)}
            infer_sig = {"extract": extract_sig,
                         "inference": fingerprint(deps["inference"]),
                         "config": fingerprint(deps["config"]),
                         "weight": fingerprint(deps["weight"]),
                         "skyseg": fingerprint(deps["skyseg"]),
                         "conf_thres": 50.0, "predict_motion": True}
            visual_sig = {"inference": infer_sig, "visualize": fingerprint(deps["visualize"]),
                          "opacity": 0.85, "gamma": 1.0, "threshold": 0.0,
                          "align": "more-crop"}
            signatures = {"extract": extract_sig, "inference": infer_sig,
                          "visualize": visual_sig}

            def stage(name: str, target: Path, verify, work) -> None:
                start = time.perf_counter()
                try:
                    exists = target.exists() or target.is_symlink()
                    if exists:
                        ensure(resume, f"目标目录已存在，不会覆盖: {target}；可使用 --resume")
                        ensure(target.is_dir() and not target.is_symlink(),
                               f"目标路径不是正常目录: {target}")
                        ensure(state.get(name) == signatures[name],
                               f"{name} 状态签名不匹配/缺失，不能安全续跑: {target}")
                        verify(target)
                        record[name + "_skipped"] = True
                        print(f"[SKIP] {name}: {target}", flush=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temp = target.parent / f".{target.name}.partial.{run_id}"
                        ensure(not temp.exists(), f"临时目录意外存在: {temp}")
                        print(f"[START] {name}: {target}", flush=True)
                        try:
                            work(temp)
                            verify(temp)
                        except Exception:
                            if temp.exists():
                                print(f"[DIAG] 失败的临时目录保留供排查: {temp}", flush=True)
                            raise
                        ensure(not target.exists(), f"目标目录在运行期间被创建: {target}")
                        temp.rename(target)
                        state[name] = signatures[name]
                        write_json(state_file, state)
                        print(f"[DONE] {name}: {target}", flush=True)
                finally:
                    record[name + "_s"] = round(time.perf_counter() - start, 3)

            stage("extract", image_dir,
                  lambda p: verify_extract(p, count),
                  lambda p: extract(video, p, count, info["total"],
                                    deps["code"], log_dir / "extract.log"))
            ensure(fingerprint(video) == manifest["video"], "源视频在抽帧过程中发生变化")
            stage("inference", inference_dir,
                  lambda p: verify_inference(p, count),
                  lambda p: infer(image_dir, p, deps["code"], log_dir / "inference.log"))
            stage("visualize", overlay_dir,
                  lambda p: verify_overlay(p, count),
                  lambda p: overlay(image_dir, inference_dir, p, deps["code"],
                                    log_dir / "visualize.log"))
            ensure(fingerprint(video) == manifest["video"], "源视频在运行过程中发生变化")
            record["status"] = "OK"
            return 0
    except KeyboardInterrupt:
        record["status"] = "INTERRUPTED"
        record["error"] = "用户终止 Ctrl+C"
        print("\n[ERROR] 用户终止", file=sys.stderr, flush=True)
        return 130
    except (PipelineError, OSError, ValueError, KeyError, SyntaxError,
            subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        record["error"] = str(exc)
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        record["total_s"] = round(time.perf_counter() - begun, 3)
        write_json(summary, record)
        print("[SUMMARY] " + json.dumps(record, ensure_ascii=False), flush=True)


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("必须 >= 1")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="单个 MP4 视频路径")
    parser.add_argument("--frames", type=positive_integer, required=True,
                        help="目标采样帧数 N，均匀覆盖原视频，最多提取源视频帧数")
    parser.add_argument("--root", type=Path, default=Path("/workspace/MoRe"))
    parser.add_argument("--mode", choices=("check", "run"), default="run")
    parser.add_argument("--resume", action="store_true", help="校验已有输出后断点续跑")
    args = parser.parse_args()
    return process(args.root, args.video, args.frames, args.mode, args.resume)


if __name__ == "__main__":
    sys.exit(main())
