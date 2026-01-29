"""
fno_pretrained_ns2d_infer_and_export.py

Pretrained FNO inference (no training) on a standard 2D Navier–Stokes vorticity-style dataset,
then export to MATLAB-friendly .mat (channel-first arrays like FlowBench).

Sources for model+dataset repo IDs/filenames follow the HF Space pattern. :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
from huggingface_hub import hf_hub_download


OUT_DIR = Path("ns2d_pretrained_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HF repos/filenames
HF_SPACE_REPO_ID = "ajsbsd/Navier_Stokes"
HF_CKPT_FILENAME = "fno_ckpt_single_res"
HF_DATASET_REPO_ID = "ajsbsd/navier-stokes-2d-dataset"
HF_DATASET_FILENAME = "navier_stokes_2d.pt"

# Inference/export settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NSAMPLES_EXPORT = 32
PLOT_ONE_SAMPLE = True
PLOT_INDEX = 0

def safe_torch_load(path: str | Path, map_location: str = "cpu"):
    path = str(path)
    try:
        return torch.load(path, weights_only=False, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_initial_conditions_tensor(dataset_path: str | Path) -> torch.Tensor:
    data = safe_torch_load(dataset_path, map_location="cpu")
    if isinstance(data, dict) and "x" in data:
        x = data["x"]
    elif isinstance(data, torch.Tensor):
        x = data
    else:
        raise ValueError(f"Unknown dataset format at {dataset_path}. Keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape [N,H,W]. Got {tuple(x.shape)}")
    return x


def vorticity_to_velocity_fft(
    omega: np.ndarray, Lx: float = 1.0, Ly: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given 2D vorticity omega(y,x), compute streamfunction psi and velocity (u,v) for periodic box:
        omega = d v/dx - d u/dy = -Δ psi
        u = d psi/dy
        v = -d psi/dx

    Solve in Fourier domain:
        psi_hat = omega_hat / k^2   (with k=0 mode set to 0)
    """
    if omega.ndim != 2:
        raise ValueError(f"omega must be 2D [Ny,Nx]. Got {omega.shape}")

    Ny, Nx = omega.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)  # [Ny,Nx]
    K2 = KX**2 + KY**2

    omega_hat = np.fft.fft2(omega)
    denom = np.where(K2 == 0.0, 1.0, K2)
    psi_hat = omega_hat / denom
    psi_hat[0, 0] = 0.0

    u_hat = 1j * KY * psi_hat
    v_hat = -1j * KX * psi_hat

    psi = np.fft.ifft2(psi_hat).real
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real
    return psi, u, v


def make_xy_grid(Ny: int, Nx: int, Lx: float = 1.0, Ly: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    return x, y


def main() -> None:
    print(f"[info] DEVICE = {DEVICE}")

    ckpt_path = hf_hub_download(
        repo_id=HF_SPACE_REPO_ID,
        filename=HF_CKPT_FILENAME,
        repo_type="space",
    )

    data_path = hf_hub_download(repo_id=HF_DATASET_REPO_ID, filename=HF_DATASET_FILENAME)

    print(f"[info] ckpt_path = {ckpt_path}")
    print(f"[info] data_path = {data_path}")

    # Load model
    model = safe_torch_load(ckpt_path, map_location=DEVICE)
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"Expected model to be torch.nn.Module, got {type(model)}")
    model.eval()
    model.to(DEVICE)

    # Load dataset
    x0_all = load_initial_conditions_tensor(data_path)  # CPU tensor
    N, H, W = x0_all.shape
    print(f"[info] dataset x shape = {tuple(x0_all.shape)}")

    n_export = min(NSAMPLES_EXPORT, N)
    idx = np.random.choice(N, size=n_export, replace=False)
    idx.sort()

    X_data = np.zeros((n_export, 3, H, W), dtype=np.float32)
    Y_data = np.zeros((n_export, 4, H, W), dtype=np.float32)

    omega0_store = np.zeros((n_export, H, W), dtype=np.float32)
    omega_pred_store = np.zeros((n_export, H, W), dtype=np.float32)

    # Grid + mask
    x, y = make_xy_grid(H, W, Lx=1.0, Ly=1.0)
    mask = np.ones((H, W), dtype=np.float32)

    # Inference loop
    with torch.no_grad():
        for j, i in enumerate(idx):
            omega0 = x0_all[i].numpy().astype(np.float32)

            inp = x0_all[i : i + 1].unsqueeze(1).to(DEVICE)
            pred = model(inp)
            pred = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            if pred.shape != (H, W):
                raise ValueError(f"Unexpected pred shape {pred.shape}, expected {(H,W)}")

            # Convert to velocity
            _, u0, v0 = vorticity_to_velocity_fft(omega0, Lx=1.0, Ly=1.0)
            _, up, vp = vorticity_to_velocity_fft(pred, Lx=1.0, Ly=1.0)
            speedp = np.sqrt(up**2 + vp**2).astype(np.float32)

            X_data[j, 0] = omega0
            X_data[j, 1] = u0.astype(np.float32)
            X_data[j, 2] = v0.astype(np.float32)

            Y_data[j, 0] = up.astype(np.float32)
            Y_data[j, 1] = vp.astype(np.float32)
            Y_data[j, 2] = pred
            Y_data[j, 3] = speedp

            omega0_store[j] = omega0
            omega_pred_store[j] = pred

    # Export .mat
    out_mat = OUT_DIR / "ns2d_fno_pretrained_export.mat"
    savemat(
        out_mat,
        {
            "X_data": X_data,
            "Y_data": Y_data,
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "mask": mask,
            "sample_idx": idx.astype(np.int32),
            # extras (handy for debugging/plots)
            "omega0": omega0_store,
            "omega_pred": omega_pred_store,
        },
        do_compression=True,
    )
    print(f"[ok] wrote {out_mat.resolve()}")

    # Quick plot
    if PLOT_ONE_SAMPLE:
        j = min(PLOT_INDEX, n_export - 1)
        om0 = omega0_store[j]
        omp = omega_pred_store[j]
        up = Y_data[j, 0]
        vp = Y_data[j, 1]
        sp = Y_data[j, 3]

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(om0, origin="lower")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f"Initial vorticity ω₀ (sample {int(idx[j])})")

        ax2 = plt.subplot(2, 2, 2)
        im2 = ax2.imshow(omp, origin="lower")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("Predicted vorticity ω̂")

        ax3 = plt.subplot(2, 2, 3)
        im3 = ax3.imshow(sp, origin="lower")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("Predicted speed |û|")

        ax4 = plt.subplot(2, 2, 4)
        ax4.imshow(sp, origin="lower")
        ax4.set_title("Velocity quiver on |û|")
        step = max(1, H // 16)
        yy, xx = np.mgrid[0:H:step, 0:W:step]
        ax4.quiver(xx, yy, up[::step, ::step], vp[::step, ::step], scale=40.0, color="white", alpha=0.9)

        plt.tight_layout()
        fig_path = OUT_DIR / "quicklook.png"
        plt.savefig(fig_path, dpi=200)
        print(f"[ok] wrote {fig_path.resolve()}")
        plt.show()


if __name__ == "__main__":
    main()