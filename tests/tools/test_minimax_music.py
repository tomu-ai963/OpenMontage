"""Tests for the fal.ai MiniMax Music 2.6 tool. No network: requests is faked."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.audio.minimax_music import COST_PER_GENERATION_USD, FAL_ENDPOINT, MiniMaxMusic
from tools.base_tool import ToolStatus

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
PROMPT = "Slow ambient meditation pad with soft singing bowls"


def test_identity_and_registry_fields():
    info = MiniMaxMusic().get_info()
    assert info["name"] == "minimax_music"
    assert info["capability"] == "music_generation"
    assert info["provider"] == "minimax"
    assert "generate_background_music" in MiniMaxMusic.capabilities
    assert MiniMaxMusic.supports["instrumental"] is True


def test_status_follows_fal_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    assert MiniMaxMusic().get_status() == ToolStatus.UNAVAILABLE
    monkeypatch.setenv("FAL_KEY", "test-key")
    assert MiniMaxMusic().get_status() == ToolStatus.AVAILABLE


def test_cost_is_flat_per_generation():
    tool = MiniMaxMusic()
    assert tool.estimate_cost({"prompt": PROMPT}) == COST_PER_GENERATION_USD
    assert tool.estimate_cost({"prompt": PROMPT, "duration_seconds": 300}) == COST_PER_GENERATION_USD


@pytest.mark.parametrize(
    "inputs, fragment",
    [
        ({"prompt": "short"}, "prompt must be"),
        ({"prompt": "x" * 2001}, "prompt must be"),
        ({"prompt": PROMPT, "is_instrumental": False}, "provide lyrics"),
        ({"prompt": PROMPT, "lyrics": "x" * 3501}, "lyrics must be"),
        ({"prompt": PROMPT, "format": "flac"}, "format must be"),
        ({"prompt": PROMPT, "duration_seconds": 0}, "duration_seconds"),
    ],
)
def test_validate_inputs_rejects(inputs, fragment):
    assert fragment in (MiniMaxMusic.validate_inputs(inputs) or "")


def test_validate_inputs_accepts_song_with_optimizer():
    assert MiniMaxMusic.validate_inputs(
        {"prompt": PROMPT, "is_instrumental": False, "lyrics_optimizer": True}
    ) is None


def test_payload_instrumental_drops_lyrics():
    payload = MiniMaxMusic.build_payload({"prompt": PROMPT, "lyrics": "la la"})
    assert payload["is_instrumental"] is True
    assert "lyrics" not in payload
    assert payload["audio_setting"]["format"] == "mp3"
    # The API rejects string values with 422 literal_error.
    assert payload["audio_setting"]["sample_rate"] == 44100
    assert payload["audio_setting"]["bitrate"] == 256000


def test_payload_song_keeps_lyrics():
    payload = MiniMaxMusic.build_payload({"prompt": PROMPT, "is_instrumental": False, "lyrics": "la la"})
    assert payload["is_instrumental"] is False
    assert payload["lyrics"] == "la la"


def test_execute_without_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    result = MiniMaxMusic().execute({"prompt": PROMPT})
    assert not result.success
    assert "FAL_KEY" in (result.error or "")


def _sine_mp3(path: Path, seconds: int) -> bytes:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-c:a", "libmp3lame", "-b:a", "128k", str(path)],
        check=True,
    )
    return path.read_bytes()


class _Resp:
    def __init__(self, json_data=None, content=b"", status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def _fake_fal(monkeypatch, audio_bytes: bytes, calls: list):
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url, json))
        return _Resp({"status_url": "https://q/status", "response_url": "https://q/result"})

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url, None))
        if url.endswith("/status"):
            return _Resp({"status": "COMPLETED"})
        if url.endswith("/result"):
            return _Resp({"audio": {"url": "https://cdn/audio.mp3"}})
        return _Resp(content=audio_bytes)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("tools.audio.minimax_music.time.sleep", lambda s: None)
    monkeypatch.setenv("FAL_KEY", "test-key")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_execute_trims_to_duration_and_keeps_raw(monkeypatch, tmp_path):
    calls: list = []
    _fake_fal(monkeypatch, _sine_mp3(tmp_path / "src.mp3", 40), calls)
    out = tmp_path / "music" / "bed.mp3"

    result = MiniMaxMusic().execute({"prompt": PROMPT, "duration_seconds": 30, "output_path": str(out)})

    assert result.success, result.error
    assert calls[0][0] == "POST" and calls[0][1].endswith(FAL_ENDPOINT)
    assert calls[0][2]["is_instrumental"] is True
    assert result.data["raw_duration_seconds"] == pytest.approx(40, abs=0.2)
    assert result.data["duration_seconds"] == pytest.approx(30, abs=0.2)
    assert (tmp_path / "music" / "bed_raw.mp3").exists()
    assert result.cost_usd == COST_PER_GENERATION_USD


def test_execute_reports_validation_422(monkeypatch, tmp_path):
    import requests

    _fake_fal(monkeypatch, b"", [])
    detail = {"detail": [{"type": "literal_error", "loc": ["body", "audio_setting", "sample_rate"]}]}

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/status"):
            return _Resp({"status": "COMPLETED"})
        return _Resp(detail, status_code=422)

    monkeypatch.setattr(requests, "get", fake_get)
    result = MiniMaxMusic().execute({"prompt": PROMPT, "output_path": str(tmp_path / "x.mp3")})

    assert not result.success
    assert "422" in (result.error or "") and "literal_error" in (result.error or "")
    assert "billed" not in (result.error or "")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_execute_short_track_is_not_padded(monkeypatch, tmp_path):
    _fake_fal(monkeypatch, _sine_mp3(tmp_path / "src.mp3", 10), [])
    out = tmp_path / "bed.mp3"

    result = MiniMaxMusic().execute({"prompt": PROMPT, "duration_seconds": 30, "output_path": str(out)})

    assert result.success, result.error
    assert result.data["duration_seconds"] == pytest.approx(10, abs=0.2)
    assert any("shorter than requested" in n for n in result.data["notes"])


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_execute_without_duration_keeps_full_track(monkeypatch, tmp_path):
    _fake_fal(monkeypatch, _sine_mp3(tmp_path / "src.mp3", 12), [])
    out = tmp_path / "full.mp3"

    result = MiniMaxMusic().execute({"prompt": PROMPT, "output_path": str(out)})

    assert result.success, result.error
    assert result.artifacts == [str(out)]
    assert result.data["duration_seconds"] == pytest.approx(12, abs=0.2)
