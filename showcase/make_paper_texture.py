# ruff: noqa: E501
"""Generate seamless paper-texture tiles for the showcase page.

Neutral by construction: every pixel is pure black or pure white at a small alpha,
so the texture adds tooth, mottling and fibres without tinting the paper colour.

    python -m showcase.make_paper_texture
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

N = 512
OUT = Path(__file__).resolve().parent / "app" / "static" / "textures"


def periodic_noise(rng, scale: float) -> np.ndarray:
    """Tileable noise: white noise low-passed in the Fourier domain (periodic by construction)."""
    white = rng.standard_normal((N, N))
    fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2)
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * np.exp(-(radius * scale) ** 2)))
    return (out - out.mean()) / (out.std() + 1e-9)


def fibres(rng, count: int) -> np.ndarray:
    """Short, slightly curved fibres drawn with wrap-around so the tile stays seamless."""
    img = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(img)
    for _ in range(count):
        x, y = rng.uniform(0, N, 2)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(10, 46)
        bend = rng.uniform(-0.35, 0.35)
        pts = []
        for t in np.linspace(0, 1, 8):
            a = ang + bend * t
            pts.append((x + np.cos(a) * length * t, y + np.sin(a) * length * t))
        shade = int(rng.uniform(40, 120))
        for dx in (-N, 0, N):
            for dy in (-N, 0, N):
                d.line([(px + dx, py + dy) for px, py in pts], fill=shade, width=1)
    return np.asarray(img, dtype=np.float32) / 255.0


def tile(strength: float, seed: int = 7, mottle_w: float = 0.0, fibre_count: int = 0) -> Image.Image:
    """Default is plain office (A4) paper: an even sheet with only a fine, faint tooth.
    mottle_w / fibre_count add cloudiness and fibres for a handmade-paper look."""
    rng = np.random.default_rng(seed)
    tooth = periodic_noise(rng, 1.2)                     # fine, very slightly soft grain
    signal = tooth + periodic_noise(rng, 90) * mottle_w
    signal = signal / np.abs(signal).max()
    fib = fibres(rng, fibre_count) if fibre_count else 0.0
    light = np.clip(signal, 0, None)                     # lighter specks
    dark = np.clip(-signal, 0, None) + fib * 0.9         # darker specks (+ fibres)
    alpha = np.clip((light + dark) * strength, 0, 1)
    white = (light > dark).astype(np.float32)
    rgba = np.zeros((N, N, 4), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = (white * 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tile(0.045).save(OUT / "paper-light.png", optimize=True)
    tile(0.04, seed=11).save(OUT / "paper-dark.png", optimize=True)
    for f in sorted(OUT.glob("*.png")):
        print(f, f"{f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
