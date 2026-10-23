#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay MoRe motion masks on the corresponding source RGB frames.

Pairing: lexicographically sorted input images <-> mask_00000.png, mask_00001.png, ...
           matching the sorted(...) logic in MoRe inference.py.

By default restores masks from MoRe's load_and_preprocess_images(..., mode='crop')
geometry to source-image resolution, including per-frame center crop and
multi-frame center padding. Areas removed by center cropping are left unchanged.

Dependencies: numpy, Pillow, opencv-python (already installed in MoRe environment).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MASK_RE = re.compile(r'^mask_(\d+)\.png$', re.IGNORECASE)
TARGET = 518
PATCH = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Overlay MoRe 8-bit Motion Mask as intensity-weighted green on original images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image-dir', required=True, type=Path, help='原始图像目录')
    parser.add_argument('--mask-dir', required=True, type=Path, help='MoRe 输出的 masks 目录')
    parser.add_argument('--output-dir', required=True, type=Path, help='可视化结果输出目录')
    parser.add_argument('--opacity', type=float, default=0.85,
                        help='最大绿色叠加透明度，范围 [0,1]')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='低于此运动响应的像素不叠加绿色，范围 [0,1]')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='强度映射指数，>1 压低弱响应，<1 提升弱响应，须大于 0')
    parser.add_argument('--align', choices=('more-crop', 'stretch', 'exact'), default='more-crop',
                        help='more-crop: 按 MoRe 默认预处理逆映射；stretch: 直接缩放；exact: 要求尺寸一致')
    return parser.parse_args()


def get_inputs(image_dir: Path, mask_dir: Path) -> tuple[list[Path], list[Path]]:
    if not image_dir.is_dir():
        raise ValueError(f'原图目录不存在: {image_dir}')
    if not mask_dir.is_dir():
        raise ValueError(f'Motion Mask 目录不存在: {mask_dir}')

    # MoRe inference.py uses sorted(path strings), not natural sorting.
    images = sorted((p for p in image_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: p.name)
    mask_dict: dict[int, Path] = {}
    for p in mask_dir.iterdir():
        if not p.is_file():
            continue
        match = MASK_RE.fullmatch(p.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx in mask_dict:
            raise ValueError(f'重复的 Mask 序号 {idx}: {mask_dict[idx]} 与 {p}')
        mask_dict[idx] = p

    if not images:
        raise ValueError(f'原图目录内未找到 jpg/jpeg/png: {image_dir}')
    if not mask_dict:
        raise ValueError(f'未找到 mask_00000.png 等文件: {mask_dir}')

    expected = set(range(len(images)))
    actual = set(mask_dict)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f'原图数量 {len(images)}，Mask 数量 {len(mask_dict)}，且 Mask 序号不连续或不匹配；'
            f'缺少的序号: {missing[:20]}，多出的序号: {unexpected[:20]}。'
            '为防止错配，已停止。'
        )

    masks = [mask_dict[i] for i in range(len(images))]
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('存在不同扩展名但同名的原图；输出将发生文件覆盖，请先处理重名文件')
    return images, masks


def load_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            if im.mode == 'RGBA':
                background = Image.new('RGBA', im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(background, im)
            return np.asarray(im.convert('RGB'), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f'无法读取原图 {path}: {exc}') from exc


def load_mask(path: Path) -> np.ndarray:
    # OpenCV IMREAD_UNCHANGED enforces the expected uint8/2D output format.
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'无法读取 Motion Mask: {path}')
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f'期望 8-bit 单通道 PNG，但 {path.name} 是 shape={mask.shape}, dtype={mask.dtype}')
    return mask.astype(np.float32) / 255.0


def preprocess_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    """MoRe 'crop' geometry: width=518, height rounded to multiples of 14.

    Returns (processed_width, processed_height, resized_height, crop_y).
    """
    resized_height = round((height * (TARGET / width)) / PATCH) * PATCH
    if resized_height <= 0:
        raise ValueError(f'原始尺寸过于极端，MoRe 处理后高度非正: {width}x{height}')
    crop_y = (resized_height - TARGET) // 2 if resized_height > TARGET else 0
    return TARGET, min(resized_height, TARGET), resized_height, crop_y


def restore_more_crop(mask: np.ndarray, original_width: int, original_height: int,
                      batch_h: int, batch_w: int) -> np.ndarray:
    """Reverse MoRe crop path. No predictions are invented outside cropped FOV."""
    processed_w, processed_h, resized_h, crop_y = preprocess_geometry(original_width, original_height)
    got_h, got_w = mask.shape
    if (got_h, got_w) != (batch_h, batch_w):
        raise ValueError(
            f'Mask 形状 {mask.shape} 与按所有原图计算的批量尺寸 {(batch_h, batch_w)} 不一致；'
            '可能推理时使用了其他预处理流程或输入子集。可以明确指定 --align stretch，'
            '但这不能保证空间位置正确。'
        )

    # MoRe centers each frame in the largest model-space batch canvas.
    pad_top = (batch_h - processed_h) // 2
    pad_left = (batch_w - processed_w) // 2
    roi = mask[pad_top:pad_top + processed_h, pad_left:pad_left + processed_w]
    if roi.shape != (processed_h, processed_w):
        raise ValueError(f'批量图像补边逆变换失败: {roi.shape}')

    if resized_h > TARGET:
        # Image was vertically center-cropped; mark out-of-FOV as transparent.
        canvas = np.zeros((resized_h, TARGET), dtype=np.float32)
        canvas[crop_y:crop_y + TARGET, :] = roi
        restored = cv2.resize(canvas, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        # Suppress tiny alpha leakage outside the modeled region.
        visibility = np.zeros((resized_h, TARGET), dtype=np.uint8)
        visibility[crop_y:crop_y + TARGET] = 1
        visibility = cv2.resize(visibility, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        restored[visibility == 0] = 0.0
        return np.clip(restored, 0, 1)

    # No crop: model observed the full original image (up to resizing/quantization).
    return np.clip(cv2.resize(roi, (original_width, original_height),
                              interpolation=cv2.INTER_LINEAR), 0, 1)


def paint_green(image: np.ndarray, mask: np.ndarray, opacity: float,
                gamma: float, threshold: float) -> np.ndarray:
    if mask.shape != image.shape[:2]:
        raise ValueError(f'待叠加图像尺寸 {image.shape[:2]} 与 mask {mask.shape} 不一致')
    intensity = np.clip(mask, 0.0, 1.0)
    alpha = (np.power(intensity, gamma) * opacity).astype(np.float32)
    alpha[intensity < threshold] = 0.0
    # RGB green=(0,255,0); alpha varies with motion intensity, not a binary color map.
    target_green = np.array((0.0, 255.0, 0.0), dtype=np.float32)
    result = image.astype(np.float32) * (1.0 - alpha[..., None]) + target_green * alpha[..., None]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def run() -> None:
    args = parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError('--opacity 必须在 [0,1] 范围内')
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError('--threshold 必须在 [0,1] 范围内')
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError('--gamma 必须是大于 0 的有限数')

    image_dir = args.image_dir.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir in (image_dir, mask_dir):
        raise ValueError('输出目录不能与原图目录或 mask 目录相同，避免覆盖源文件')

    images, masks = get_inputs(image_dir, mask_dir)
    if args.align == 'more-crop':
        sizes = []
        for path in images:
            with Image.open(path) as im:
                sizes.append(preprocess_geometry(*im.size))
        batch_h = max(h for _, h, _, _ in sizes)
        batch_w = max(w for w, _, _, _ in sizes)
    else:
        batch_h = batch_w = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = output_dir / 'pairing.csv'
    records = []
    n = len(images)
    for i, (im_path, mask_path) in enumerate(zip(images, masks)):
        rgb = load_image(im_path)
        response = load_mask(mask_path)
        h, w = rgb.shape[:2]
        if args.align == 'more-crop':
            response = restore_more_crop(response, w, h, batch_h, batch_w)
        elif args.align == 'stretch':
            response = cv2.resize(response, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            if response.shape != (h, w):
                raise ValueError(f'--align exact: {mask_path.name} 的尺寸 {response.shape} 与原图 {(h,w)} 不一致')
        out_name = f'{im_path.stem}_motion_overlay.png'
        out_path = output_dir / out_name
        overlay = paint_green(rgb, response, args.opacity, args.gamma, args.threshold)
        Image.fromarray(overlay, mode='RGB').save(out_path, format='PNG')
        records.append((i, im_path.name, mask_path.name, out_name, w, h))
        print(f'[{i+1:>3}/{n}] {im_path.name} + {mask_path.name} -> {out_name}')

    # Manifest is written only after processing all pairs successfully.
    with pairs_csv.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(('index', 'image_name', 'mask_name', 'output_name', 'output_width', 'output_height'))
        writer.writerows(records)
    print(f'\n完成：{n} 帧，输出目录：{output_dir}')
    print(f'配对明细：{pairs_csv}')


if __name__ == '__main__':
    try:
        run()
    except (ValueError, OSError, cv2.error) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
