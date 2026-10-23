#!/usr/bin/env python3
"""MoRe single-video pipeline: native-frame extraction -> inference -> mask overlay."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def file_info(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def probe_video(video: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
        "-of", "json", str(video),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
    require(proc.returncode == 0, f"ffprobe 失败: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    require(len(streams) == 1, "视频必须包含可读的第一个视频流")
    stream = streams[0]
    count = int(stream.get("nb_read_frames") or 0)
    require(count > 0, "ffprobe 无法统计视频帧数，拒绝在无法核验完整性的情况下继续")
    avg = stream.get("avg_frame_rate", "0/0")
    nominal = stream.get("r_frame_rate", "0/0")
    def positive_rate(rate: str) -> bool:
        try:
            return Fraction(rate) > 0
        except (ValueError, ZeroDivisionError):
            return False
    fps_str = avg if positive_rate(avg) else nominal
    require(positive_rate(fps_str), "无效的视频帧率")
    require(int(stream.get("width", 0)) > 0 and int(stream.get("height", 0)) > 0,
            "视频尺寸无效")
    return {
        "frames": count,
        "fps": fps_str,
        "nominal_fps": nominal,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name", "unknown"),
    }


def check_files(root: Path, video: Path) -> dict:
    for executable in ("ffmpeg", "ffprobe"):
        require(shutil.which(executable) is not None, f"找不到 {executable}；请先安装 FFmpeg")
    require(video.is_file(), f"视频不存在: {video}")
    require(video.stat().st_size > 0, f"视频为空: {video}")
    code = root / "code" / "MoRe"
    paths = {
        "code": code,
        "inference": code / "inference.py",
        "visualize": code / "test" / "visualize_more_motion_mask.py",
        "config": code / "training" / "config" / "omniworld_full.yaml",
        "checkpoint": code / "pretrained" / "more_full.pt",
        "skyseg": code / "pretrained" / "skyseg.onnx",
    }
    require(code.is_dir(), f"代码目录不存在: {code}")
    for key, path in paths.items():
        if key != "code":
            require(path.is_file() and path.stat().st_size > 0, f"缺少文件 {key}: {path}")
    for key in ("inference", "visualize"):
        script = paths[key]
        try:
            compile(script.read_bytes(), str(script), "exec")
        except SyntaxError as exc:
            raise PipelineError(f"脚本语法检查失败: {script}: {exc}") from exc
    return paths


def files_at(directory: Path, pattern: re.Pattern) -> list[Path]:
    require(directory.is_dir(), f"目录不存在: {directory}")
    matched = []
    for path in directory.iterdir():
        if path.is_file() and pattern.fullmatch(path.name):
            matched.append(path)
    return sorted(matched)


def verify_files(directory: Path, pattern: re.Pattern, expected: int, kind: str) -> None:
    paths = files_at(directory, pattern)
    require(len(paths) == expected,
            f"{kind} 数量错误: {directory}；期望 {expected}，实际 {len(paths)}")
    require(all(p.stat().st_size > 0 for p in paths), f"{kind} 存在空文件: {directory}")
    if kind in ("视频帧", "motion mask", "逐帧 PLY", "depth NPY",
                "extrinsic NPY", "intrinsic NPY"):
        indexes = sorted(int(pattern.fullmatch(p.name).group(1)) for p in paths)
        require(all(value == i for i, value in enumerate(indexes)),
                f"{kind} 编号不连续/不从0开始: {directory}")


FRAME_RE = re.compile(r"frame_(\d+)\.png", re.IGNORECASE)
MASK_RE = re.compile(r"mask_(\d+)\.png", re.IGNORECASE)
OVERLAY_RE = re.compile(r"[^.].*\.(png|jpg|jpeg)", re.IGNORECASE)
PLY_RE = re.compile(r"frame_(\d+)\.ply", re.IGNORECASE)
NPY_RE = re.compile(r"(\d+)\.npy", re.IGNORECASE)


def verify_inference(directory: Path, expected: int) -> None:
    verify_files(directory / "masks", MASK_RE, expected, "motion mask")
    verify_files(directory, PLY_RE, expected, "逐帧 PLY")
    for subdir in ("depth", "extrinsic", "intrinsic"):
        verify_files(directory / subdir, NPY_RE, expected, subdir + " NPY")


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("CMD:", " ".join(map(str, command)), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                stream.write(line)
                print(line, end="", flush=True)
            code = proc.wait()
    require(code == 0, f"命令失败，退出码={code}；日志: {log}")


def extract(video: Path, output: Path, root: Path, log: Path, limit: int | None) -> None:
    require(not output.exists(), f"临时输出目录已存在: {output}")
    output.mkdir(parents=True)
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(video), "-map", "0:v:0", "-fps_mode", "passthrough",
        "-start_number", "0", "-compression_level", "3",
    ]
    if limit is not None:
        command += ["-frames:v", str(limit)]
    command += [str(output / "frame_%08d.png")]
    run_logged(command, root, log)


def inference(image_dir: Path, output: Path, paths: dict, log: Path) -> None:
    require(not output.exists(), f"临时输出目录已存在: {output}")
    cmd = [
        sys.executable, "inference.py",
        "--config_path", "training/config/omniworld_full.yaml",
        "--ckpt_path", "pretrained/more_full.pt",
        "--image_path", str(image_dir),
        "--output_dir", str(output),
        "--skyseg_onnx", "pretrained/skyseg.onnx",
        "--conf_thres", "50.0", "--predict_motion",
    ]
    run_logged(cmd, paths["code"], log)


def visualize(image_dir: Path, mask_dir: Path, output: Path, paths: dict, log: Path) -> None:
    require(not output.exists(), f"临时输出目录已存在: {output}")
    cmd = [
        sys.executable, "test/visualize_more_motion_mask.py",
        "--image-dir", str(image_dir),
        "--mask-dir", str(mask_dir),
        "--output-dir", str(output),
        "--opacity", "0.85", "--gamma", "1.0",
        "--threshold", "0.0", "--align", "more-crop",
    ]
    run_logged(cmd, paths["code"], log)


def signatures(video: Path, info: dict, paths: dict) -> dict:
    source = {"video": file_info(video), "metadata": info, "format": "png", "fps_mode": "passthrough"}
    model = {
        "source": source,
        "inference": file_info(paths["inference"]),
        "config": file_info(paths["config"]),
        "checkpoint": file_info(paths["checkpoint"]),
        "skyseg": file_info(paths["skyseg"]),
        "conf_thres": 50.0,
        "predict_motion": True,
    }
    overlay = {
        "model": model,
        "visualize": file_info(paths["visualize"]),
        "opacity": 0.85, "gamma": 1.0, "threshold": 0.0, "align": "more-crop",
    }
    return {"extract": source, "inference": model, "visualize": overlay}


def pipeline(video: Path, root: Path, mode: str, resume: bool, smoke_frames: int,
             summary_path: Path | None) -> int:
    video = video.resolve()
    root = root.resolve()
    stem = video.stem
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    log_dir = root / "results" / "pipeline_logs" / stem / run_id
    result = {
        "video": str(video), "sequence": stem, "mode": mode,
        "status": "FAILED", "frames": None, "fps": None,
        "extract_s": None, "inference_s": None, "visualize_s": None,
        "extract_skipped": False, "inference_skipped": False,
        "visualize_skipped": False, "error": "",
        "log_dir": str(log_dir),
    }
    started = time.perf_counter()
    summary = summary_path.resolve() if summary_path else (log_dir / "summary.json")
    try:
        paths = check_files(root, video)
        info = probe_video(video)
        result["frames"] = info["frames"]
        result["fps"] = info["fps"]
        print(f"[CHECK] {video.name}: {info['frames']} frames, "
              f"{info['width']}x{info['height']}, avg_fps={info['fps']}, "
              f"r_fps={info['nominal_fps']}", flush=True)
        if mode == "check":
            result["status"] = "OK"
            return 0
        n = min(smoke_frames, info["frames"]) if mode == "smoke" else info["frames"]
        sigs = signatures(video, info, paths)
        state_file = root / "results" / "pipeline_state" / (stem + ".json")
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
        if mode == "smoke":
            (root / "results").mkdir(parents=True, exist_ok=True)
            smoke_base = Path(tempfile.mkdtemp(prefix=f".smoke_{stem}_", dir=root / "results"))
            print(f"[SMOKE] 测试目录: {smoke_base}，抽取前 {n} 帧", flush=True)
            image_dir = smoke_base / "frames"
            infer_dir = smoke_base / "inference"
            overlay_dir = smoke_base / "overlay"
        else:
            smoke_base = None
            image_dir = root / "data-tiny" / stem
            infer_dir = root / "results" / f"{stem}_full_glbfix"
            overlay_dir = root / "results" / f"{stem}_motion_overlay"

        def stage(name: str, target: Path, verify, work) -> None:
            key = name + "_s"
            start = time.perf_counter()
            try:
                if target.exists():
                    require(mode == "run" and resume,
                            f"输出目录已存在（禁止覆盖）: {target}；确认完整后可用 --resume")
                    require(state.get(name) == sigs[name],
                            f"{name} 的输入/参数与已有输出不匹配或缺少状态记录: {target}")
                    verify(target)
                    result[name + "_skipped"] = True
                    print(f"[SKIP] {name}: {target}", flush=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = (target.parent / f".{target.name}.partial.{run_id}") if mode == "run" else target
                    require(not tmp.exists(), f"存在旧临时目录: {tmp}")
                    print(f"[START] {name}: {target}", flush=True)
                    work(tmp)
                    verify(tmp)
                    if mode == "run":
                        # output and temporary directory are siblings: same filesystem, atomic rename
                        tmp.rename(target)
                        state[name] = sigs[name]
                        atomic_json(state_file, state)
                    print(f"[DONE] {name}: {target}", flush=True)
            finally:
                result[key] = round(time.perf_counter() - start, 3)

        stage("extract", image_dir,
              lambda p: verify_files(p, FRAME_RE, n, "视频帧"),
              lambda p: extract(video, p, root, log_dir / "extract.log",
                                n if mode == "smoke" else None))
        require(file_info(video) == sigs["extract"]["video"], "原始视频在处理中被修改")
        stage("inference", infer_dir,
              lambda p: verify_inference(p, n),
              lambda p: inference(image_dir, p, paths, log_dir / "inference.log"))
        stage("visualize", overlay_dir,
              lambda p: verify_files(p, OVERLAY_RE, n, "可视化图片"),
              lambda p: visualize(image_dir, infer_dir / "masks", p, paths,
                                  log_dir / "visualize.log"))
        require(file_info(video) == sigs["extract"]["video"], "原始视频在推理期间被修改")
        result["status"] = "OK"
        return 0
    except (PipelineError, OSError, ValueError, json.JSONDecodeError,
            subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        print(f"[ERROR] {video.name}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        result["total_s"] = round(time.perf_counter() - started, 3)
        # Keep temporary smoke artifacts if any stage fails; otherwise clean them.
        if "smoke_base" in locals() and smoke_base is not None:
            if result["status"] == "OK":
                try:
                    shutil.rmtree(smoke_base)
                except OSError as exc:
                    print(f"[WARN] 无法清理 smoke 临时目录: {exc}", file=sys.stderr)
            else:
                result["smoke_artifacts"] = str(smoke_base)
        atomic_json(summary, result)
        print("[SUMMARY] " + json.dumps(result, ensure_ascii=False), flush=True)


@contextmanager
def sequence_lock(root: Path, stem: str):
    """Prevent two processes from writing the same sequence concurrently."""
    lock_dir = root / "results" / "pipeline_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / (stem + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"该序列已有另一个处理进程正在运行: {stem}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="源 MP4 视频")
    parser.add_argument("--root", type=Path, default=Path("/workspace/MoRe"))
    parser.add_argument("--mode", choices=("check", "smoke", "run"), default="run")
    parser.add_argument("--smoke-frames", type=int, default=32,
                        help="smoke 检验使用的前 N 帧（默认 32）")
    parser.add_argument("--resume", action="store_true",
                        help="仅跳过带匹配状态记录且帧数完整的既有输出")
    parser.add_argument("--summary-json", type=Path,
                        help="结果 JSON 的保存位置；默认自动保存至 pipeline_logs")
    args = parser.parse_args()
    parser.error("--smoke-frames 必须 >= 1") if args.smoke_frames < 1 else None
    try:
        with sequence_lock(args.root.resolve(), args.video.stem):
            return pipeline(args.video, args.root, args.mode, args.resume,
                            args.smoke_frames, args.summary_json)
    except (PipelineError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
