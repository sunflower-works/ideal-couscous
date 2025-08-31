#!/usr/bin/env python3
"""Watermark algorithms module.

Provides:
  - Abstract base interface BaseWatermarker
  - DCTParityWatermarker: existing 8x8 block DCT mid-band parity embedding. (legacy baseline)
  - LSBWatermarker: simple spatial-domain LSB (legacy fragile algorithm).
  - DWTSVDWatermarker / DWTSVDAdvWatermarker: Haar DWT + SVD singular-value parity embedding.
  - DWTAdaptiveWatermarker: Adaptive DWT-SVD variant with rank-based quant step modulation.

Exports a registry WATERMARKERS for algorithm selection and helper functions
for distinguishing mainstream vs legacy algorithms.
"""
from __future__ import annotations

import hashlib
import math
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Protocol

import numpy as np
from PIL import Image

# ------------------- DCT support -------------------
_C = None
_DEF_N = 8


def _dct_matrix(n: int = 8) -> np.ndarray:
    global _C
    if _C is not None and _C.shape[0] == n:
        return _C
    C = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            C[k, i] = alpha * math.cos(((2 * i + 1) * k * math.pi) / (2 * n))
    _C = C
    return _C


def dct2(block: np.ndarray) -> np.ndarray:
    C = _dct_matrix(block.shape[0])
    return C @ block @ C.T


def idct2(coeff: np.ndarray) -> np.ndarray:
    C = _dct_matrix(coeff.shape[0])
    return C.T @ coeff @ C


# ------------------- Shared types -------------------
MID_BAND_COORDS: List[Tuple[int, int]] = [
    (2, 3),
    (3, 2),
    (3, 3),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
]


@dataclass
class WatermarkResult:
    stego: Image.Image
    extracted_bits: List[int]
    embed_time_ms: float
    extract_time_ms: float
    psnr_db: float
    ssim: float
    ber_percent: float


# Metrics


def psnr(orig: np.ndarray, stego: np.ndarray) -> float:
    mse = np.mean((orig - stego) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10 * math.log10((255.0**2) / mse)


def ssim_global(orig: np.ndarray, stego: np.ndarray) -> float:
    o = orig.astype(np.float64)
    s = stego.astype(np.float64)
    mu_o = o.mean()
    mu_s = s.mean()
    sigma_o = o.var()
    sigma_s = s.var()
    sigma_os = ((o - mu_o) * (s - mu_s)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    return ((2 * mu_o * mu_s + C1) * (2 * sigma_os + C2)) / (
        (mu_o**2 + mu_s**2 + C1) * (sigma_o + sigma_s + C2)
    )


# ------------------- Interface -------------------
class BaseWatermarker(Protocol):
    name: str

    def embed(
        self, img: Image.Image, payload_bits: Iterable[int], key: str
    ) -> WatermarkResult: ...
    def extract(self, img: Image.Image, payload_len: int, key: str) -> List[int]: ...


# ------------------- DCT Parity Implementation -------------------


def _derive_prng_indices(total_blocks: int, needed: int, key: bytes) -> List[int]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    idxs = np.arange(total_blocks)
    rng.shuffle(idxs)
    return idxs[:needed].tolist()


class DCTParityWatermarker:
    name = "dct_parity"
    mainstream = False  # legacy baseline

    def __init__(self, coords: List[Tuple[int, int]] | None = None):
        self.coords = coords or MID_BAND_COORDS

    def embed(
        self, img: Image.Image, payload_bits: Iterable[int], key: str
    ) -> WatermarkResult:
        payload = list(payload_bits)
        start = time.perf_counter()
        rgb = img.convert("RGB")
        YCbCr = rgb.convert("YCbCr")
        Y, Cb, Cr = [np.array(ch, dtype=np.float32) for ch in YCbCr.split()]
        h, w = Y.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            Y = np.pad(Y, ((0, pad_h), (0, pad_w)), mode="edge")
        blocks_h = Y.shape[0] // 8
        blocks_w = Y.shape[1] // 8
        total_blocks = blocks_h * blocks_w
        needed_blocks = math.ceil(len(payload) / len(self.coords))
        if needed_blocks > total_blocks:
            raise ValueError("Payload exceeds image DCT capacity")
        block_order = _derive_prng_indices(total_blocks, needed_blocks, key.encode())
        Y_work = Y.copy()
        bit_i = 0
        for bidx in block_order:
            if bit_i >= len(payload):
                break
            bh = bidx // blocks_w
            bw = bidx % blocks_w
            block = Y_work[bh * 8 : (bh + 1) * 8, bw * 8 : (bw + 1) * 8]
            coeff = dct2(block - 128.0)
            for r, cx in self.coords:
                if bit_i >= len(payload):
                    break
                val = coeff[r, cx]
                mag = abs(val)
                mag_int = int(round(mag))
                desired = payload[bit_i]
                if mag_int % 2 != desired:
                    mag_int += 1
                coeff[r, cx] = math.copysign(mag_int, val)
                bit_i += 1
            block_rec = idct2(coeff) + 128.0
            Y_work[bh * 8 : (bh + 1) * 8, bw * 8 : (bw + 1) * 8] = np.clip(
                block_rec, 0, 255
            )
        Y_final = Y_work[:h, :w].astype(np.uint8)
        stego = Image.merge(
            "YCbCr",
            [
                Image.fromarray(Y_final),
                Image.fromarray(Cb.astype(np.uint8)),
                Image.fromarray(Cr.astype(np.uint8)),
            ],
        ).convert("RGB")
        embed_ms = (time.perf_counter() - start) * 1000.0
        t2 = time.perf_counter()
        extracted = self.extract(stego, len(payload), key)
        extract_ms = (time.perf_counter() - t2) * 1000.0
        orig_Y = np.array(img.convert("L"), dtype=np.float32)
        stego_Y = np.array(stego.convert("L"), dtype=np.float32)
        psnr_db = psnr(orig_Y, stego_Y)
        ssim_val = ssim_global(orig_Y, stego_Y)
        diff = sum(1 for a, b in zip(payload, extracted) if a != b)
        ber_percent = 100.0 * diff / len(payload) if payload else 0.0
        return WatermarkResult(
            stego, extracted, embed_ms, extract_ms, psnr_db, ssim_val, ber_percent
        )

    def extract(self, img: Image.Image, payload_len: int, key: str) -> List[int]:
        rgb = img.convert("RGB")
        Y = np.array(rgb.convert("L"), dtype=np.float32)
        h, w = Y.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            Y = np.pad(Y, ((0, pad_h), (0, pad_w)), mode="edge")
        blocks_h = Y.shape[0] // 8
        blocks_w = Y.shape[1] // 8
        total_blocks = blocks_h * blocks_w
        needed_blocks = math.ceil(payload_len / len(self.coords))
        block_order = _derive_prng_indices(total_blocks, needed_blocks, key.encode())
        bits: List[int] = []
        bit_i = 0
        for bidx in block_order:
            if bit_i >= payload_len:
                break
            bh = bidx // blocks_w
            bw = bidx % blocks_w
            block = Y[bh * 8 : (bh + 1) * 8, bw * 8 : (bw + 1) * 8]
            coeff = dct2(block - 128.0)
            for r, cx in self.coords:
                if bit_i >= payload_len:
                    break
                mag_int = int(round(abs(coeff[r, cx])))
                bits.append(mag_int % 2)
                bit_i += 1
        return bits


# ------------------- LSB Implementation -------------------
class LSBWatermarker:
    name = "lsb"
    mainstream = False  # explicitly legacy

    def embed(
        self, img: Image.Image, payload_bits: Iterable[int], key: str
    ) -> WatermarkResult:
            "LSBWatermarker is legacy and fragile to compression attacks; not recommended for robust use.",
            RuntimeWarning,
        )
        payload = list(payload_bits)
        start = time.perf_counter()
        arr = np.array(img.convert("L"))
        flat = arr.flatten()
        if len(payload) > len(flat):
            raise ValueError("Payload exceeds pixel capacity")
        # Embed
        flat[: len(payload)] = (flat[: len(payload)] & 0xFE) | np.array(
            payload, dtype=np.uint8
        )
        stego_arr = flat.reshape(arr.shape)
        stego_img = Image.fromarray(stego_arr).convert("RGB")
        embed_ms = (time.perf_counter() - start) * 1000.0
        t2 = time.perf_counter()
        extracted = self.extract(stego_img, len(payload), key)
        extract_ms = (time.perf_counter() - t2) * 1000.0
        orig = arr.astype(np.float32)
        stego_f = stego_arr.astype(np.float32)
        psnr_db = psnr(orig, stego_f)
        ssim_val = ssim_global(orig, stego_f)
        diff = sum(1 for a, b in zip(payload, extracted) if a != b)
        ber_percent = 100.0 * diff / len(payload) if payload else 0.0
        return WatermarkResult(
            stego_img, extracted, embed_ms, extract_ms, psnr_db, ssim_val, ber_percent
        )

    def extract(self, img: Image.Image, payload_len: int, key: str) -> List[int]:
        arr = np.array(img.convert("L"))
        flat = arr.flatten()
        bits = (flat[:payload_len] & 1).astype(np.uint8).tolist()
        return bits


# ------------------- DWT + SVD Implementation (prototype) -------------------


def _haar_dwt2(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-level 2D Haar DWT. Pads to even dims by edge replication."""
    h, w = arr.shape
    if h % 2 == 1:
        arr = np.pad(arr, ((0, 1), (0, 0)), mode="edge")
        h += 1
    if w % 2 == 1:
        arr = np.pad(arr, ((0, 0), (0, 1)), mode="edge")
        w += 1
    # Rows
    avg_rows = (arr[:, 0::2] + arr[:, 1::2]) / 2.0
    dif_rows = (arr[:, 0::2] - arr[:, 1::2]) / 2.0
    # Columns
    LL = (avg_rows[0::2, :] + avg_rows[1::2, :]) / 2.0
    LH = (avg_rows[0::2, :] - avg_rows[1::2, :]) / 2.0
    HL = (dif_rows[0::2, :] + dif_rows[1::2, :]) / 2.0
    HH = (dif_rows[0::2, :] - dif_rows[1::2, :]) / 2.0
    return LL, LH, HL, HH


def _haar_idwt2(
    LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray
) -> np.ndarray:
    # Inverse of above forward transform.
    avg_rows_top = LL + LH
    avg_rows_bot = LL - LH
    dif_rows_top = HL + HH
    dif_rows_bot = HL - HH
    avg_rows = np.empty(
        (avg_rows_top.shape[0] * 2, avg_rows_top.shape[1]), dtype=LL.dtype
    )
    dif_rows = np.empty_like(avg_rows)
    avg_rows[0::2, :] = avg_rows_top
    avg_rows[1::2, :] = avg_rows_bot
    dif_rows[0::2, :] = dif_rows_top
    dif_rows[1::2, :] = dif_rows_bot
    rec = np.empty((avg_rows.shape[0], avg_rows.shape[1] * 2), dtype=LL.dtype)
    rec[:, 0::2] = avg_rows + dif_rows
    rec[:, 1::2] = avg_rows - dif_rows
    return rec


class DWTSVDWatermarker:
    """Embed bits by adjusting parity of quantized singular value indices of LL subband.
    Uses a uniform quantization step `q` then enforces (floor(|S|/q) % 2)==bit by adding +/- q.
    Capacity: up to len(S) bits (number of singular values)."""

    name = "dwt_svd"
    mainstream = True

    def __init__(self, q: float = 4.0):
        if q <= 0:
            raise ValueError("q must be > 0")
        self.q = q

    def embed(
        self, img: Image.Image, payload_bits: Iterable[int], key: str
    ) -> WatermarkResult:
        payload = list(payload_bits)
        start = time.perf_counter()
        gray = np.array(img.convert("L"), dtype=np.float32)
        LL, LH, HL, HH = _haar_dwt2(gray)
        U, S, Vt = np.linalg.svd(LL, full_matrices=False)
        if len(payload) > len(S):
            raise ValueError("Payload exceeds DWT-SVD capacity")
        S_mod = S.copy()
        q = self.q
        for i, bit in enumerate(payload):
            sval = S_mod[i]
            sign = 1.0 if sval >= 0 else -1.0
            mag = abs(sval)
            q_index = int(math.floor(mag / q))
            if (q_index % 2) != bit:
                # shift minimally to flip parity
                mag = (q_index + 1) * q + (mag - q_index * q)
            S_mod[i] = sign * mag
        LL_mod = (U * S_mod) @ Vt
        rec = _haar_idwt2(LL_mod, LH, HL, HH)
        rec = np.clip(rec, 0, 255)
        stego_img = Image.fromarray(rec.astype(np.uint8)).convert("RGB")
        embed_ms = (time.perf_counter() - start) * 1000.0
        t2 = time.perf_counter()
        extracted = self.extract(stego_img, len(payload), key)
        extract_ms = (time.perf_counter() - t2) * 1000.0
        orig_f = gray
        stego_f = np.array(stego_img.convert("L"), dtype=np.float32)
        psnr_db = psnr(orig_f, stego_f)
        ssim_val = ssim_global(orig_f, stego_f)
        diff = sum(1 for a, b in zip(payload, extracted) if a != b)
        ber_percent = 100.0 * diff / len(payload) if payload else 0.0
        return WatermarkResult(
            stego_img, extracted, embed_ms, extract_ms, psnr_db, ssim_val, ber_percent
        )

    def extract(self, img: Image.Image, payload_len: int, key: str) -> List[int]:
        gray = np.array(img.convert("L"), dtype=np.float32)
        LL, _, _, _ = _haar_dwt2(gray)
        _, S, _ = np.linalg.svd(LL, full_matrices=False)
        if payload_len > len(S):
            payload_len = len(S)
        q = self.q
        bits = []
        for i in range(payload_len):
            mag = abs(S[i])
            q_index = int(math.floor(mag / q))
            bits.append(q_index % 2)
        return bits


class DWTSVDAdvWatermarker(DWTSVDWatermarker):
    """Placeholder advanced variant (would include adversarial optimisation).
    Currently identical to DWTSVDWatermarker for prototype wiring."""

    name = "dwt_svd_adv"
    mainstream = True

    def __init__(self, q: float = 4.0):
        super().__init__(q=q)


class DWTAdaptiveWatermarker(DWTSVDWatermarker):
    """Adaptive variant of DWT-SVD: scales quantization step per singular value rank.

    q_i = base_q * (1 + adapt_alpha * (1 - rank_norm)) where rank_norm in [0,1].
    Earlier (stronger energy) singular values get slightly larger steps to improve robustness,
    later values smaller steps to preserve imperceptibility. Parity embedding same as parent.
    """

    name = "dwt_svd_adapt"
    mainstream = True

    def __init__(self, q: float = 4.0, adapt_alpha: float = 0.35):
        super().__init__(q=q)
        self.adapt_alpha = adapt_alpha

    def embed(
        self, img: Image.Image, payload_bits: Iterable[int], key: str
    ) -> WatermarkResult:
        payload = list(payload_bits)
        start = time.perf_counter()
        gray = np.array(img.convert("L"), dtype=np.float32)
        LL, LH, HL, HH = _haar_dwt2(gray)
        U, S, Vt = np.linalg.svd(LL, full_matrices=False)
        if len(payload) > len(S):
            raise ValueError("Payload exceeds DWT-SVD capacity")
        S_mod = S.copy()
        n = len(S_mod)
        base_q = self.q
        for i, bit in enumerate(payload):
            sval = S_mod[i]
            sign = 1.0 if sval >= 0 else -1.0
            mag = abs(sval)
            rank_norm = i / max(1, n - 1)
            q_i = base_q * (1 + self.adapt_alpha * (1 - rank_norm))
            q_index = int(math.floor(mag / q_i))
            if (q_index % 2) != bit:
                mag = (q_index + 1) * q_i + (mag - q_index * q_i)
            S_mod[i] = sign * mag
        LL_mod = (U * S_mod) @ Vt
        rec = _haar_idwt2(LL_mod, LH, HL, HH)
        rec = np.clip(rec, 0, 255)
        stego_img = Image.fromarray(rec.astype(np.uint8)).convert("RGB")
        embed_ms = (time.perf_counter() - start) * 1000.0
        t2 = time.perf_counter()
        extracted = self.extract(stego_img, len(payload), key)
        extract_ms = (time.perf_counter() - t2) * 1000.0
        orig_f = gray
        stego_f = np.array(stego_img.convert("L"), dtype=np.float32)
        psnr_db = psnr(orig_f, stego_f)
        ssim_val = ssim_global(orig_f, stego_f)
        diff = sum(1 for a, b in zip(payload, extracted) if a != b)
        ber_percent = 100.0 * diff / len(payload) if payload else 0.0
        return WatermarkResult(
            stego_img, extracted, embed_ms, extract_ms, psnr_db, ssim_val, ber_percent
        )


# ------------------- Registry -------------------
WATERMARKERS: Dict[str, BaseWatermarker] = {
    "dct_parity": DCTParityWatermarker(),
    "lsb": LSBWatermarker(),
    "dwt_svd": DWTSVDWatermarker(q=4.0),
    "dwt_svd_adv": DWTSVDAdvWatermarker(q=4.0),
    "dwt_svd_adapt": DWTAdaptiveWatermarker(q=4.0, adapt_alpha=0.35),
}


def list_mainstream() -> List[str]:
    return [k for k, v in WATERMARKERS.items() if getattr(v, "mainstream", False)]


__all__ = [
    "BaseWatermarker",
    "DCTParityWatermarker",
    "LSBWatermarker",
    "DWTSVDWatermarker",
    "DWTSVDAdvWatermarker",
    "DWTAdaptiveWatermarker",
    "list_mainstream",
]
