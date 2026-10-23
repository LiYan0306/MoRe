#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove MoRe-predicted dynamic pixels from original RGB frames.

For each source frame, pixels whose stored 8-bit MoRe motion response / 255
is STRICTLY GREATER THAN --threshold are replaced with black (RGB 0,0,0).
All other *decoded* RGB pixel values remain unchanged. A binary dynamic mask
(white=removed, black=retained) is saved alongside the masked images by default.

Pairing follows the MoRe inference image ordering: lexicographically sorted
filenames, paired with mask_00000.png, mask_00001.png, ... in index order.

--align more-crop inversely maps the default MoRe "crop" preprocessing:
width=518, resized height rounded to a multiple of 14, optional vertical
center crop to 518, then batch-wise center padding. Areas never seen by the
model remain unmasked. Nearest-neighbor resampling keeps binary decisions
binary. An inverse resize is approximate at subpixel borders.

Requires: Python >=3.10, numpy, Pillow, opencv-python (or opencv-python-headless).
No PyTorch/GPU required. Does not modify source images or MoRe masks.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SOURCE_EXTS = {".png", ".jpg", ".jpeg"}
MASK_NAME = re.compile(r"mask_(\d+)\.png", re.IGNORECASE)
TARGET_WIDTH = 518
PATCH_SIZE = 14


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    resized_height: int
    processed_height: int
    crop_top: int


def geometry_for_image(width: int, height: int) -> Geometry:
    """Replicate load_and_preprocess_images(..., mode='crop') shape rules."""
    if width <= 0 or height <= 0:
        raise ValueError(f"无效图像尺寸：{width}×{height}")
    resized_h = round((height * (TARGET_WIDTH / width)) / PATCH_SIZE) * PATCH_SIZE
    if resized_h < PATCH_SIZE:
        raise ValueError(f"MoRe 预处理后的高度为 {resized_h}，尺寸异常：{width}×{height}")
    processed_h = min(resized_h, TARGET_WIDTH)
    crop_top = (resized_h - TARGET_WIDTH) // 2 if resized_h > TARGET_WIDTH else 0
    return Geometry(width, height, resized_h, processed_h, crop_top)


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError("--image-dir 和 --mask-dir 必须是已存在的目录")
    # MoRe uses sorted(list of complete image paths); same parent, so sort by name.
    images = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SOURCE_EXTS),
        key=lambda p: p.name,
    )
    if not images:
        raise ValueError(f"没有找到 PNG/JPG/JPEG 输入图像：{image_dir}")

    numbered: dict[int, Path] = {}
    for path in mask_dir.iterdir():
        if not path.is_file():
            continue
        found = MASK_NAME.fullmatch(path.name)
        if found is None:
            continue
        idx = int(found.group(1))
        if idx in numbered:
            raise ValueError(f"出现重复 mask 帧号 {idx}：{numbered[idx]} 与 {path}")
        numbered[idx] = path

    expected = set(range(len(images)))
    actual = set(numbered)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"原图 {len(images)} 帧、匹配命名的 mask {len(numbered)} 帧；"
            f"缺失编号={missing[:15]}，多余编号={extra[:15]}。"
            "为防止错误配对，已终止。"
        )
    return [(image, numbered[i]) for i, image in enumerate(images)]


def read_original_rgb(path: Path) -> np.ndarray:
    # No EXIF auto-rotation: MoRe loads the untransposed pixel matrix too.
    try:
        with Image.open(path) as im:
            if im.mode == "RGBA":
                white = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(white, im)
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as e:
        raise ValueError(f"无法读取原图 {path}: {e}") from e


def read_motion_u8(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            if im.mode != "L":
                raise ValueError(
                    f"{path} 不是 8-bit 灰度图（PIL mode={im.mode}）；"
                    "预期为 MoRe 导出的 mask_*.png"
                )
            return np.asarray(im, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as e:
        raise ValueError(f"无法读取 Mask {path}: {e}") from e


def mask_in_original_coords(
    dynamic_model: np.ndarray,
    geom: Geometry,
    batch_height: int,
    align: str,
) -> np.ndarray:
    """Map a *binary* model-space dynamic mask into original image coordinates."""
    if dynamic_model.ndim != 2 or dynamic_model.dtype != np.bool_:
        raise ValueError("内部错误：模型 Mask 必须是二维 bool 数组")
    width, height = geom.width, geom.height
    if align == "exact":
        if dynamic_model.shape != (height, width):
            raise ValueError(
                f"--align exact 尺寸不一致：mask {dynamic_model.shape}，原图 {(height, width)}"
            )
        return dynamic_model
    if align == "stretch":
        return cv2.resize(
            dynamic_model.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    if align != "more-crop":
        raise ValueError(f"未知对齐方式：{align}")

    expected_shape = (batch_height, TARGET_WIDTH)
    if dynamic_model.shape != expected_shape:
        raise ValueError(
            f"mask 尺寸 {dynamic_model.shape} 与 MoRe crop 预期尺寸 {expected_shape} 不符。"
            "请确认推理输入帧集合、预处理模式是否一致；"
            "不能确认时请勿盲目使用 --align stretch。"
        )
    # Reverse batch center padding (only the source frame's valid model region).
    pad_top = (batch_height - geom.processed_height) // 2
    region = dynamic_model[pad_top:pad_top + geom.processed_height, :]
    if region.shape != (geom.processed_height, TARGET_WIDTH):
        raise ValueError("无法正确逆转批量图像的中心补边")

    if geom.resized_height > TARGET_WIDTH:
        # Original tall frames were vertically center-cropped. Unknown pixels
        # outside the model field of view must stay unchanged.
        uncropped = np.zeros((geom.resized_height, TARGET_WIDTH), dtype=np.uint8)
        top = geom.crop_top
        uncropped[top:top + TARGET_WIDTH, :] = region.astype(np.uint8)
    else:
        uncropped = region.astype(np.uint8)
    return cv2.resize(uncropped, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def write_png_atomically(array: np.ndarray, path: Path) -> None:
    """Write a complete PNG then atomically replace destination on same filesystem."""
    fd, temp_name = tempfile.mkstemp(prefix=".more_mask_tmp_", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        Image.fromarray(array).save(temp_name, format="PNG")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_manifest_atomically(rows: list[tuple], path: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".more_pairs_tmp_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow((
                "index", "source_image", "motion_mask", "masked_image",
                "binary_mask", "original_width", "original_height", "removed_pixels", "removed_ratio",
            ))
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def validate_output_paths(output_dir: Path, image_dir: Path, mask_dir: Path) -> None:
    for source in (image_dir, mask_dir):
        if output_dir == source or output_dir in source.parents or source in output_dir.parents:
            raise ValueError(
                f"输出目录不得与输入目录相同，或为其父/子目录：{output_dir} vs {source}"
            )


def process(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    threshold: float,
    align: str = "more-crop",
    save_binary: bool = True,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Return (number of frames, total number of blacked-out pixels)."""
    if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError("--threshold 必须是 [0, 1] 范围内的有限小数")
    if align not in {"more-crop", "stretch", "exact"}:
        raise ValueError("--align 必须为 more-crop / stretch / exact")

    image_dir = image_dir.expanduser().resolve()
    mask_dir = mask_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    validate_output_paths(output_dir, image_dir, mask_dir)
    pairs = list_pairs(image_dir, mask_dir)

    # Compute the preprocessed batch shape using *all* source images, like MoRe.
    geometries: list[Geometry] = []
    for image_path, _ in pairs:
        try:
            with Image.open(image_path) as im:
                geometries.append(geometry_for_image(*im.size))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"无法读取原图尺寸 {image_path}: {exc}") from exc
    batch_height = max(g.processed_height for g in geometries)

    image_out = output_dir / "images"
    mask_out = output_dir / "dynamic_masks"
    jobs: list[tuple[Path, Path, Path, Path, Geometry]] = []
    all_paths = [output_dir / "pairing.csv"]
    for index, ((img, mmask), geom) in enumerate(zip(pairs, geometries)):
        stem = f"{index:05d}_{img.stem}"
        dest_image = image_out / f"{stem}_motion_removed.png"
        dest_binary = mask_out / f"{stem}_dynamic_mask.png"
        jobs.append((img, mmask, dest_image, dest_binary, geom))
        all_paths.append(dest_image)
        if save_binary:
            all_paths.append(dest_binary)

    if not overwrite:
        existing = [str(p) for p in all_paths if p.exists()]
        if existing:
            raise ValueError(
                "输出文件已存在，默认拒绝覆盖；请另选 --output-dir，"
                "或明确添加 --overwrite。冲突文件示例：" + ", ".join(existing[:5])
            )

    # Validate all masks before writing *any* new output file. This catches
    # index/shape mismatches early (including mixed-size batch padding).
    for img, mask_file, _, _, geom in jobs:
        dynamic_u8 = read_motion_u8(mask_file)
        if align == "more-crop" and dynamic_u8.shape != (batch_height, TARGET_WIDTH):
            raise ValueError(
                f"{mask_file.name} 尺寸 {dynamic_u8.shape} 与预期 {(batch_height, TARGET_WIDTH)} 不符"
            )
        if align == "exact" and dynamic_u8.shape != (geom.height, geom.width):
            raise ValueError(
                f"{mask_file.name} 尺寸 {dynamic_u8.shape} 与 {img.name} 尺寸 "
                f"{(geom.height, geom.width)} 不符"
            )

    image_out.mkdir(parents=True, exist_ok=True)
    if save_binary:
        mask_out.mkdir(parents=True, exist_ok=True)
    rows: list[tuple] = []
    total_removed = 0
    for index, (img_file, mask_file, dest_img, dest_bin, geom) in enumerate(jobs):
        original = read_original_rgb(img_file)
        mask = read_motion_u8(mask_file)
        # Quantized value v represents v/255. Strictly greater than threshold.
        # Compare via float64 to avoid the boundary drift of float32 conversion.
        dynamic = mask.astype(np.float64) > (threshold * 255.0)
        dynamic = mask_in_original_coords(dynamic, geom, batch_height, align)
        if dynamic.shape != original.shape[:2]:
            raise ValueError(f"映射后的 Mask 形状不正确：{dynamic.shape}")
        result = original.copy()
        result[dynamic] = (0, 0, 0)
        write_png_atomically(result, dest_img)
        if save_binary:
            write_png_atomically(dynamic.astype(np.uint8) * 255, dest_bin)
        n_removed = int(np.count_nonzero(dynamic))
        total_removed += n_removed
        rel_bin = str(dest_bin.relative_to(output_dir)) if save_binary else ""
        rows.append((
            index, img_file.name, mask_file.name, str(dest_img.relative_to(output_dir)),
            rel_bin, geom.width, geom.height, n_removed,
            f"{n_removed / (geom.width * geom.height):.8f}",
        ))
        print(f"[{index + 1:>3}/{len(jobs)}] {img_file.name} ← {mask_file.name} "
              f"| dynamic {n_removed / (geom.width * geom.height):.2%}")
    write_manifest_atomically(rows, output_dir / "pairing.csv")
    return len(jobs), total_removed


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将 MoRe 预测为动态的像素置黑，其余像素严格保留原图 RGB 值",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-dir", required=True, type=Path, help="原图文件夹")
    p.add_argument("--mask-dir", required=True, type=Path, help="MoRe motion mask 文件夹")
    p.add_argument("--output-dir", required=True, type=Path, help="输出文件夹")
    p.add_argument("--threshold", required=True, type=float,
                   help="0~1 浮点数，mask/255 > threshold 的位置变黑（严格大于）")
    p.add_argument("--align", default="more-crop", choices=("more-crop", "stretch", "exact"),
                   help="more-crop 还原 MoRe 默认预处理；stretch 强制拉伸；exact 要求同尺寸")
    p.add_argument("--save-binary-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="是否同时保存二值动态区域掩码：255=被置黑，0=保留")
    p.add_argument("--overwrite", action="store_true", help="明确允许覆盖本程序对应的输出文件")
    return p


def main() -> int:
    args = make_parser().parse_args()
    try:
        frames, black_pixels = process(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            align=args.align,
            save_binary=args.save_binary_mask,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"\n完成：{frames} 帧，置黑像素总数={black_pixels}")
    print(f"输出目录：{args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
