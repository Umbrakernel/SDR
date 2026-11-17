#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", nargs="?", default="rx.cf32")
    p.add_argument("--rate", type=float, default=None,
                   help="sample rate, needed only when sidecar JSON is absent")
    p.add_argument("--start", type=int, default=0, help="first sample index")
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--save", help="PNG output instead of an interactive window")
    a = p.parse_args()
    path = Path(a.file)
    meta_path = Path(str(path) + ".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rate = a.rate if a.rate is not None else meta.get("sample_rate_hz", 1e6)
    if not np.isfinite(rate) or rate <= 0 or a.start < 0 or a.count < 2:
        p.error("rate must be positive; start >= 0; count >= 2")
    if not path.exists() or path.stat().st_size == 0:
        p.error("IQ file is missing or empty")
    if path.stat().st_size % 8:
        p.error("CF32 file length is not a multiple of 8 bytes")
    dtype = ">c8" if meta.get("byte_order") == "big" else "<c8"
    data = np.memmap(path, dtype=dtype, mode="r")
    z = np.asarray(data[a.start:a.start+a.count], dtype=np.complex64)
    if len(z) < 2:
        p.error("not enough samples at the requested start index")
    if a.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(z)) / rate * 1e3
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    ax[0].plot(t, z.real, label="I", linewidth=0.8)
    ax[0].plot(t, z.imag, label="Q", linewidth=0.8)
    ax[0].set(xlabel="Time from selected start, ms", ylabel="Amplitude", title=path.name)
    ax[0].legend()
    window = np.hanning(len(z))
    spectrum = np.fft.fftshift(np.fft.fft(z * window)) / max(window.sum(), 1)
    freq = np.fft.fftshift(np.fft.fftfreq(len(z), 1/rate)) / 1e3
    ax[1].plot(freq, 20*np.log10(np.maximum(np.abs(spectrum), 1e-12)))
    ax[1].set(xlabel="Baseband frequency, kHz", ylabel="Magnitude, dBFS")
    ax[2].scatter(z.real[::max(1, len(z)//5000)],
                  z.imag[::max(1, len(z)//5000)], s=2)
    ax[2].set(xlabel="I", ylabel="Q", title="IQ scatter (all sample phases)")
    ax[2].set_aspect("equal", adjustable="datalim")
    for item in ax:
        item.grid(True, alpha=0.3)
    if a.save:
        fig.savefig(a.save, dpi=150)
        print(a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
