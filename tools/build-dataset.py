#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性构建示例数据集：MNIST → 每类采样 → 重心居中 → 13×13 灰度 PNG。

输出：data/sample-digits/{train|test}/{label}/{i:03d}.png
  - train：每类 100 张（共 1000），test：每类 30 张（共 300）
  - 与 js/dataset.js 的 samplePaths() 命名约定保持一致（i 为类内序号，从 0 起）。

预处理与应用内管线（js/preprocess.js）语义一致、独立实现：
  1. 计算 28×28 灰度图的质心（重心）；
  2. 整数像素平移使质心落在画布中心 (13.5, 13.5)（四舍五入，边缘补 0）；
  3. resize((13,13), Image.BOX)（面积平均）；
  4. 存为 mode 'L' 的灰度 PNG。

仅依赖 PIL + 标准库（无 numpy）。MNIST 原始 .gz 缓存在 tools/.mnist-cache/（已 gitignore）。
"""

import gzip
import os
import random
import struct
import sys
import urllib.request

from PIL import Image

# ---------------- 常量 ----------------

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "tools", ".mnist-cache")
OUT_DIR = os.path.join(REPO_ROOT, "data", "sample-digits")

TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 30
SEED = 42
IMG_SIZE = 28
OUT_SIZE = 13


# ---------------- 下载 ----------------

def download(url, dest):
    """下载文件（已存在则跳过），打印进度。"""
    if os.path.exists(dest):
        print(f"已缓存，跳过下载：{os.path.basename(dest)}")
        return
    print(f"下载中：{url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1048576:.1f} / {total / 1048576:.1f} MB", end="", flush=True)
        print()
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------- idx 解析 ----------------

def read_idx_images(path):
    """解析 images idx3 文件，返回 (count, bytes)，每张 28×28 uint8。"""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x803:
            raise ValueError(f"{path}：magic 错误 {magic:#x}（期望 0x803）")
        if (rows, cols) != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"{path}：图片尺寸 {rows}×{cols}（期望 28×28）")
        data = f.read(count * IMG_SIZE * IMG_SIZE)
        if len(data) != count * IMG_SIZE * IMG_SIZE:
            raise ValueError(f"{path}：数据长度不足")
    return count, data


def read_idx_labels(path):
    """解析 labels idx1 文件，返回 bytes（每字节一个标签）。"""
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 0x801:
            raise ValueError(f"{path}：magic 错误 {magic:#x}（期望 0x801）")
        data = f.read(count)
        if len(data) != count:
            raise ValueError(f"{path}：数据长度不足")
    return data


# ---------------- 预处理 ----------------

def center_of_mass(pixels):
    """计算 28×28 灰度像素（bytes）的质心，返回 (cx, cy)；全黑图返回画布中心。"""
    mass = 0
    sx = 0
    sy = 0
    for y in range(IMG_SIZE):
        base = y * IMG_SIZE
        for x in range(IMG_SIZE):
            v = pixels[base + x]
            mass += v
            sx += v * x
            sy += v * y
    if mass == 0:
        return (IMG_SIZE - 1) / 2.0, (IMG_SIZE - 1) / 2.0
    return sx / mass, sy / mass


def center_and_downsample(pixels):
    """质心居中（整数平移）→ 13×13 面积平均 → 返回 PIL 'L' 图。"""
    cx, cy = center_of_mass(pixels)
    # 目标中心 (13.5, 13.5)，四舍五入取整数偏移
    dx = int(round(13.5 - cx))
    dy = int(round(13.5 - cy))

    img = Image.frombytes("L", (IMG_SIZE, IMG_SIZE), bytes(pixels))
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    # 手动计算重叠区域，避免对 paste 负坐标行为的假设
    src_x0, src_y0 = max(0, -dx), max(0, -dy)
    src_x1, src_y1 = min(IMG_SIZE, IMG_SIZE - dx), min(IMG_SIZE, IMG_SIZE - dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas.paste(img.crop((src_x0, src_y0, src_x1, src_y1)), (max(0, dx), max(0, dy)))
    return canvas.resize((OUT_SIZE, OUT_SIZE), Image.BOX)


# ---------------- 主流程 ----------------

def build_split(split, images_data, labels_data, per_class):
    rng = random.Random(SEED)
    # 按标签收集下标
    by_label = {d: [] for d in range(10)}
    for idx, lab in enumerate(labels_data):
        by_label[lab].append(idx)

    written = 0
    for label in range(10):
        candidates = by_label[label]
        if len(candidates) < per_class:
            raise ValueError(f"{split}/标签 {label}：仅 {len(candidates)} 张，不足 {per_class}")
        chosen = rng.sample(candidates, per_class)
        out_dir = os.path.join(OUT_DIR, split, str(label))
        os.makedirs(out_dir, exist_ok=True)
        for i, idx in enumerate(chosen):
            off = idx * IMG_SIZE * IMG_SIZE
            pixels = images_data[off:off + IMG_SIZE * IMG_SIZE]
            small = center_and_downsample(pixels)
            small.save(os.path.join(out_dir, f"{i:03d}.png"))
            written += 1
    print(f"{split}：写入 {written} 张")
    return written


def self_check():
    """脚本自带校验：总数、每类数量、抽查 20 张可读且为 13×13 'L'。"""
    total = 0
    for split, per_class in (("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)):
        for label in range(10):
            d = os.path.join(OUT_DIR, split, str(label))
            files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
            assert len(files) == per_class, f"{split}/{label}：{len(files)} 张（期望 {per_class}）"
            expected_names = [f"{i:03d}.png" for i in range(per_class)]
            assert files == expected_names, f"{split}/{label}：命名不连续"
            total += len(files)

    assert total == 1300, f"总数 {total}（期望 1300）"

    # 抽查 20 张：PIL 可读、13×13、mode 'L'、像素值有变化（非全黑）
    rng = random.Random(7)
    checked = 0
    for _ in range(20):
        split = rng.choice(["train", "test"])
        label = rng.randrange(10)
        per_class = TRAIN_PER_CLASS if split == "train" else TEST_PER_CLASS
        i = rng.randrange(per_class)
        p = os.path.join(OUT_DIR, split, str(label), f"{i:03d}.png")
        im = Image.open(p)
        im.load()
        assert im.size == (13, 13), f"{p}：尺寸 {im.size}"
        assert im.mode == "L", f"{p}：mode {im.mode}"
        px = list(im.getdata())
        assert max(px) > 0, f"{p}：全黑图"
        checked += 1
    print(f"自检通过：总数 {total}，抽查 {checked} 张均为 13×13 'L' 且非全黑")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in FILES:
        download(MNIST_URL + name, os.path.join(CACHE_DIR, name))

    n_train, train_images = read_idx_images(os.path.join(CACHE_DIR, FILES[0]))
    train_labels = read_idx_labels(os.path.join(CACHE_DIR, FILES[1]))
    n_test, test_images = read_idx_images(os.path.join(CACHE_DIR, FILES[2]))
    test_labels = read_idx_labels(os.path.join(CACHE_DIR, FILES[3]))
    print(f"MNIST 读取完毕：train {n_train} 张，test {n_test} 张")

    build_split("train", train_images, train_labels, TRAIN_PER_CLASS)
    build_split("test", test_images, test_labels, TEST_PER_CLASS)
    self_check()
    print("完成。")


if __name__ == "__main__":
    sys.exit(main())
