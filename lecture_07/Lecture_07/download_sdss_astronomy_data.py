#!/usr/bin/env python3
"""download_sdss_astronomy_data.py

Run this ONCE (outside MATLAB) to fetch all imagery for the MATLAB Image Processing
lecture and to build a small weakly-labeled segmentation dataset.

It uses the SDSS "image cutout" web service to retrieve JPEG cutouts around
(ra, dec) sky coordinates, then generates weak masks (SOURCE vs SKY) using a
classical image-processing heuristic.

Requirements:
  python>=3.9, numpy, pillow

Example run command:
  python3 download_sdss_astronomy_data.py --out_root ./data --n_train 220 --n_test 70
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass


def _require(pkg: str, import_name: str | None = None):
    try:
        return __import__(import_name or pkg)
    except Exception as e:
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install it with e.g. 'pip install {pkg}'.\n{e}"
        )


np = _require("numpy")
PIL = _require("Pillow", "PIL")
from PIL import Image, ImageFilter

BASE_URLS = [
    "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg",
    "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg",
    "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg",
]


@dataclass(frozen=True)
class Target:
    name: str
    ra: float
    dec: float


MAIN_TARGETS = [
    Target("M51_Whirlpool", 202.467917, 47.198333),
    Target("M87_VirgoA", 187.705833, 12.391111),
    Target("M101_Pinwheel", 210.804167, 54.348056),
    Target("SDSS_field", 224.594100, -1.090000),
]


def build_url(base: str, *, ra: float, dec: float, scale: float, width: int, height: int, opt: str = "") -> str:
    params = {
        "ra": f"{ra:.6f}",
        "dec": f"{dec:.6f}",
        "scale": f"{scale:.4f}",
        "width": str(int(width)),
        "height": str(int(height)),
        "opt": opt,
    }
    return base + "?" + urllib.parse.urlencode(params)


def fetch_jpeg_bytes(url: str, timeout_s: float = 20.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MATLAB-lecture-downloader/1.0 (+https://openai.com)",
            "Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return data


def download_cutout(
    *,
    ra: float,
    dec: float,
    scale: float,
    width: int,
    height: int,
    opt: str = "",
    retries: int = 2,
    sleep_s: float = 0.0,
) -> tuple[bytes, str]:
    """Try BASE_URLS until we get a valid JPEG."""

    last_err = None
    for base in BASE_URLS:
        url = build_url(base, ra=ra, dec=dec, scale=scale, width=width, height=height, opt=opt)
        for _ in range(retries + 1):
            try:
                data = fetch_jpeg_bytes(url)
                # JPEG magic bytes
                if len(data) < 1024 or data[:2] != b"\xff\xd8":
                    raise ValueError("Not a JPEG (or too small)")
                if sleep_s > 0:
                    time.sleep(sleep_s)
                return data, url
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Failed to download a valid JPEG. Last error: {last_err}")


def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    x = np.clip(x, 0.0, 1.0)
    hist, bin_edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)

    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


def _connected_components(mask: np.ndarray) -> list[dict]:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    comps = []

    nbrs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = 1
                coords = []
                rs = 0.0
                cs = 0.0
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    rs += rr
                    cs += cc
                    for dr, dc in nbrs:
                        r2 = rr + dr
                        c2 = cc + dc
                        if 0 <= r2 < H and 0 <= c2 < W and mask[r2, c2] and not visited[r2, c2]:
                            visited[r2, c2] = 1
                            stack.append((r2, c2))
                area = len(coords)
                centroid = (rs / area, cs / area)
                comps.append({"area": area, "coords": coords, "centroid": centroid})

    return comps


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    inv = (~mask).astype(np.uint8)
    seen = np.zeros((H, W), dtype=np.uint8)
    stack = []

    for c in range(W):
        if inv[0, c] and not seen[0, c]:
            stack.append((0, c)); seen[0, c] = 1
        if inv[H - 1, c] and not seen[H - 1, c]:
            stack.append((H - 1, c)); seen[H - 1, c] = 1
    for r in range(H):
        if inv[r, 0] and not seen[r, 0]:
            stack.append((r, 0)); seen[r, 0] = 1
        if inv[r, W - 1] and not seen[r, W - 1]:
            stack.append((r, W - 1)); seen[r, W - 1] = 1

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in nbrs:
            r2 = r + dr
            c2 = c + dc
            if 0 <= r2 < H and 0 <= c2 < W and inv[r2, c2] and not seen[r2, c2]:
                seen[r2, c2] = 1
                stack.append((r2, c2))

    holes = (inv == 1) & (seen == 0)
    out = mask.copy()
    out[holes] = True
    return out


def weak_source_mask(rgb: Image.Image) -> np.ndarray:
    
    g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    bg = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float32) / 255.0
    sub = np.clip(g - bg, 0.0, 1.0)

    p99 = float(np.percentile(sub, 99.0))
    if p99 > 1e-6:
        sub = np.clip(sub / p99, 0.0, 1.0)

    t = otsu_threshold(sub)
    mask = sub > max(t, 0.18)

    m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    m_img = m_img.filter(ImageFilter.MaxFilter(size=3))
    m_img = m_img.filter(ImageFilter.MinFilter(size=3))
    mask = (np.asarray(m_img) > 0)

    mask = _fill_holes(mask)

    comps = _connected_components(mask)
    if not comps:
        return np.zeros_like(mask, dtype=bool)

    H, W = mask.shape
    cr = (H - 1) / 2.0
    cc = (W - 1) / 2.0
    best = None
    best_score = -1.0

    for comp in comps:
        area = float(comp["area"])
        rmean, cmean = comp["centroid"]
        dist2 = (rmean - cr) ** 2 + (cmean - cc) ** 2
        score = area / (1.0 + dist2)
        if score > best_score:
            best_score = score
            best = comp

    out = np.zeros((H, W), dtype=bool)
    for r, c in best["coords"]:
        out[r, c] = True

    out_img = Image.fromarray((out.astype(np.uint8) * 255), mode="L")
    out_img = out_img.filter(ImageFilter.MaxFilter(size=3))
    out = (np.asarray(out_img) > 0)

    return out


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def build_seg_dataset(
    out_root: str,
    n_train: int,
    n_test: int,
    seed: int,
    sleep_s: float,
    raw_size: int = 128,
    raw_scale: float = 0.40,
    proc_size: int = 32,
) -> None:
    rng = random.Random(seed)

    seg_root = os.path.join(out_root, "sdssSeg")
    raw_train = os.path.join(seg_root, "raw", "train")
    raw_test = os.path.join(seg_root, "raw", "test")

    train_imgs = os.path.join(seg_root, "trainingImages")
    train_labs = os.path.join(seg_root, "trainingLabels")
    test_imgs = os.path.join(seg_root, "testImages")
    test_labs = os.path.join(seg_root, "testLabels")

    for p in [raw_train, raw_test, train_imgs, train_labs, test_imgs, test_labs]:
        ensure_dir(p)

    def sample_coord() -> tuple[float, float]:
        tgt = rng.choice(MAIN_TARGETS)
        off = 0.015  # ~54 arcsec
        dra = rng.uniform(-off, off)
        ddec = rng.uniform(-off, off)
        return tgt.ra + dra, tgt.dec + ddec

    def make_examples(split: str, n_needed: int) -> None:
        assert split in ("train", "test")
        raw_dir = raw_train if split == "train" else raw_test
        img_dir = train_imgs if split == "train" else test_imgs
        lab_dir = train_labs if split == "train" else test_labs

        kept = 0
        attempts = 0
        max_attempts = max(200, int(n_needed * 12))

        while kept < n_needed and attempts < max_attempts:
            attempts += 1
            ra, dec = sample_coord()

            try:
                data, url = download_cutout(
                    ra=ra,
                    dec=dec,
                    scale=raw_scale,
                    width=raw_size,
                    height=raw_size,
                    opt="",
                    sleep_s=sleep_s,
                )
            except Exception as e:
                print(f"[{split}] download failed (attempt {attempts}/{max_attempts}): {e}")
                continue

            try:
                rgb = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                print(f"[{split}] PIL decode failed: {e}")
                continue

            mask = weak_source_mask(rgb)
            frac = float(mask.mean())

            if not (0.01 <= frac <= 0.65):
                continue

            kept += 1

            raw_path = os.path.join(raw_dir, f"{kept:04d}.jpg")
            save_bytes(raw_path, data)

            g = rgb.convert("L").resize((proc_size, proc_size), resample=Image.BILINEAR)
            g_path = os.path.join(img_dir, f"{kept:04d}.png")
            g.save(g_path)

            m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            m_img = m_img.resize((proc_size, proc_size), resample=Image.NEAREST)
            m_path = os.path.join(lab_dir, f"{kept:04d}.png")
            m_img.save(m_path)

            if kept % 25 == 0 or kept == n_needed:
                print(f"[{split}] kept {kept}/{n_needed}")

        if kept < n_needed:
            raise RuntimeError(
                f"Could only build {kept}/{n_needed} {split} examples. "
                f"Try increasing --max_attempts (not exposed) by re-running, "
                f"or reduce n_train/n_test."
            )

    print("[seg] building weak segmentation dataset (SOURCE vs SKY)")
    make_examples("train", n_train)
    make_examples("test", n_test)

    ra, dec = MAIN_TARGETS[0].ra, MAIN_TARGETS[0].dec
    data, _ = download_cutout(ra=ra, dec=dec, scale=0.25, width=256, height=256, opt="", sleep_s=sleep_s)
    save_bytes(os.path.join(seg_root, "sourceTest.jpg"), data)


def download_main_images(out_root: str, sleep_s: float) -> None:
    sdss_dir = os.path.join(out_root, "sdss")
    ensure_dir(sdss_dir)

    scale = 0.25
    W = 512
    H = 512

    files = ["01.jpg", "02.jpg", "03.jpg", "04.jpg"]

    for fname, tgt in zip(files, MAIN_TARGETS):
        outpath = os.path.join(sdss_dir, fname)
        if os.path.isfile(outpath) and os.path.getsize(outpath) > 8000:
            continue
        data, url = download_cutout(ra=tgt.ra, dec=tgt.dec, scale=scale, width=W, height=H, opt="", sleep_s=sleep_s)
        save_bytes(outpath, data)
        print(f"[main] {fname} <- {tgt.name} ({tgt.ra:.3f}, {tgt.dec:.3f})")
        print(f"       {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data", help="Output root directory (default: ./data)")
    ap.add_argument("--n_train", type=int, default=220, help="#training samples for segmentation")
    ap.add_argument("--n_test", type=int, default=70, help="#test samples for segmentation")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (polite throttling)")
    ap.add_argument("--skip_seg", action="store_true", help="Only download main images (skip segmentation dataset)")
    args = ap.parse_args()

    out_root = args.out_root
    ensure_dir(out_root)

    print(f"[info] writing to: {os.path.abspath(out_root)}")
    download_main_images(out_root, sleep_s=args.sleep)

    if not args.skip_seg:
        build_seg_dataset(out_root, n_train=args.n_train, n_test=args.n_test, seed=args.seed, sleep_s=args.sleep)

    print("[done] SDSS astronomy lecture data prepared.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
