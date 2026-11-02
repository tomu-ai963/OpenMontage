"""Regression tests for audio_mixer fades and loudnorm output rate (needs ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.audio.audio_mixer import NORMALIZED_SAMPLE_RATE, AudioMixer

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on PATH"
)
SR = 48000


def _tone(path: Path, seconds: float) -> str:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:sample_rate={SR}:duration={seconds}",
         "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return str(path)


def _pcm(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def _rms(x: np.ndarray, t0: float, t1: float) -> float:
    seg = x[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def _sample_rate(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return int(out.strip())


def test_fade_out_happens_at_the_end_not_the_start(tmp_path):
    src = _tone(tmp_path / "tone.wav", 4.0)
    out = tmp_path / "mixed.wav"

    r = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": src, "role": "music", "fade_out_seconds": 1.0}],
        "normalize": False,
        "output_path": str(out),
    })

    assert r.success, r.error
    x = _pcm(out)
    full = _rms(x, 0.5, 2.5)
    assert full > 0.8 * _rms(_pcm(Path(src)), 0.5, 2.5)  # head untouched (was silenced before the fix)
    assert _rms(x, 3.0, 3.25) > 0.5 * full  # fade has started but is not yet deep
    assert _rms(x, 3.8, 4.0) < 0.2 * full   # tail is faded


def test_fade_in_follows_start_offset(tmp_path):
    src = _tone(tmp_path / "tone.wav", 3.0)
    out = tmp_path / "mixed.wav"

    r = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": src, "role": "music", "start_seconds": 2.0, "fade_in_seconds": 1.0}],
        "normalize": False,
        "output_path": str(out),
    })

    assert r.success, r.error
    x = _pcm(out)
    assert _rms(x, 0.0, 1.9) < 1e-3          # leading delay is silent
    early, late = _rms(x, 2.0, 2.2), _rms(x, 3.5, 4.5)
    assert late > 0.8 * _rms(_pcm(Path(src)), 0.5, 2.5)
    assert early < 0.5 * late                 # fade-in runs after the delay, not during it


def test_normalize_outputs_48k_not_192k(tmp_path):
    src = _tone(tmp_path / "tone.wav", 2.0)
    out = tmp_path / "norm.wav"

    r = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": src, "role": "music"}],
        "normalize": True,
        "output_path": str(out),
    })

    assert r.success, r.error
    assert _sample_rate(out) == NORMALIZED_SAMPLE_RATE == 48000
