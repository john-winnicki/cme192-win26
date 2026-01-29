"""
operator_dataset_adapter.py

A small adapter layer so your MATLAB demos can stay stable:
- standardize to channel-first tensors [N, C, Ny, Nx]
- optional: derive (u,v) from vorticity via FFT (periodic box)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from scipy.io import savemat


def vorticity_to_velocity_fft(omega: np.ndarray, Lx: float = 1.0, Ly: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if omega.ndim != 2:
        raise ValueError(f"omega must be 2D [Ny,Nx]. Got {omega.shape}")
    Ny, Nx = omega.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)
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


def export_channel_first_mat(
    out_path: str | Path,
    X_data: np.ndarray,
    Y_data: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a .mat with the same convention you’ve been using:
      X_data: [N, Cx, Ny, Nx]
      Y_data: [N, Cy, Ny, Nx]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if X_data.ndim != 4 or Y_data.ndim != 4:
        raise ValueError(f"Expected X_data/Y_data to be 4D [N,C,Ny,Nx]. Got {X_data.shape} and {Y_data.shape}")

    d = {"X_data": X_data.astype(np.float32), "Y_data": Y_data.astype(np.float32)}
    if x is not None:
        d["x"] = x.astype(np.float32)
    if y is not None:
        d["y"] = y.astype(np.float32)
    if mask is not None:
        d["mask"] = mask.astype(np.float32)
    if extra:
        d.update(extra)

    savemat(out_path, d, do_compression=True)


def omega_dataset_to_mat(
    omega0: np.ndarray,        # [N,Ny,Nx]
    omega1: np.ndarray,        # [N,Ny,Nx] (could be model pred or next-step truth)
    out_path: str | Path,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> None:
    """
    Turn an omega-only dataset into a rich (u,v,omega,|u|) dataset for MATLAB PDE visuals.

    X_data: [omega0, u0, v0]
    Y_data: [u1, v1, omega1, speed1]
    """
    if omega0.ndim != 3 or omega1.ndim != 3:
        raise ValueError("omega0/omega1 must be [N,Ny,Nx]")
    if omega0.shape != omega1.shape:
        raise ValueError(f"omega0 shape {omega0.shape} must match omega1 shape {omega1.shape}")

    N, Ny, Nx = omega0.shape
    X_data = np.zeros((N, 3, Ny, Nx), dtype=np.float32)
    Y_data = np.zeros((N, 4, Ny, Nx), dtype=np.float32)

    for i in range(N):
        _, u0, v0 = vorticity_to_velocity_fft(omega0[i], Lx=Lx, Ly=Ly)
        _, u1, v1 = vorticity_to_velocity_fft(omega1[i], Lx=Lx, Ly=Ly)
        speed1 = np.sqrt(u1**2 + v1**2)

        X_data[i, 0] = omega0[i]
        X_data[i, 1] = u0.astype(np.float32)
        X_data[i, 2] = v0.astype(np.float32)

        Y_data[i, 0] = u1.astype(np.float32)
        Y_data[i, 1] = v1.astype(np.float32)
        Y_data[i, 2] = omega1[i]
        Y_data[i, 3] = speed1.astype(np.float32)

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    mask = np.ones((Ny, Nx), dtype=np.float32)

    export_channel_first_mat(out_path, X_data, Y_data, x=x, y=y, mask=mask)

