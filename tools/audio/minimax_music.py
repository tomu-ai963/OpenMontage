"""MiniMax Music 2.6 generation via the fal.ai queue API.

Pay-as-you-go music generation (no subscription) using the same FAL_KEY as the
fal image/video tools. Supports vocal-free instrumentals via ``is_instrumental``.

The model does not take a target length, so when ``duration_seconds`` is given
the tool keeps the untouched download next to the output (``*_raw.<ext>``) and
writes a copy trimmed to that length with a fade-out (requires ffmpeg).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

FAL_ENDPOINT = "fal-ai/minimax-music/v2.6"
# Per-generation price from fal's learn page ("$0.15 per audio"). The model's API
# page shows no price, so re-check fal's pricing page if this matters.
COST_PER_GENERATION_USD = 0.15
PROMPT_MIN_CHARS = 10
PROMPT_MAX_CHARS = 2000
LYRICS_MAX_CHARS = 3500
POLL_INTERVAL_SEC = 4
POLL_TIMEOUT_SEC = 600


class MiniMaxMusic(BaseTool):
    name = "minimax_music"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "minimax"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []  # checked dynamically via FAL_KEY
    install_instructions = (
        "Set FAL_KEY to your fal.ai API key (same key as the fal image/video tools).\n"
        "  Get one at https://fal.ai/dashboard/keys and add credit at "
        "https://fal.ai/dashboard/billing"
    )
    agent_skills = ["music"]

    capabilities = [
        "generate_background_music",
        "generate_instrumental",
        "generate_song",
    ]
    supports = {
        "instrumental": True,
        "vocals": True,
        "custom_lyrics": True,
        "style_control": True,
        "exact_duration": False,  # trimmed locally with ffmpeg instead
    }
    best_for = [
        "pay-as-you-go instrumental background music (no subscription)",
        "ambient / meditation / cinematic beds trimmed to the video length",
        "songs with custom lyrics",
    ]
    not_good_for = [
        "sound effects",
        "exact-length generation (model picks the length; tool trims afterwards)",
        "offline generation",
    ]
    fallback_tools = ["music_gen", "suno_music"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": PROMPT_MIN_CHARS,
                "maxLength": PROMPT_MAX_CHARS,
                "description": "Music style, mood, genre, instruments and scenario.",
            },
            "is_instrumental": {
                "type": "boolean",
                "default": True,
                "description": "Vocal-free output. Set false for a song (then lyrics or lyrics_optimizer is required).",
            },
            "lyrics": {
                "type": "string",
                "maxLength": LYRICS_MAX_CHARS,
                "description": "Lyrics for vocal songs. Ignored when is_instrumental is true.",
            },
            "lyrics_optimizer": {
                "type": "boolean",
                "default": False,
                "description": "With is_instrumental false and no lyrics, let the model write lyrics from the prompt.",
            },
            "duration_seconds": {
                "type": "number",
                "minimum": 1,
                "description": "Trim the result to this length (seconds) with a fade-out. Omit to keep the full track.",
            },
            "fade_out_seconds": {
                "type": "number",
                "minimum": 0,
                "default": 2.0,
                "description": "Fade-out applied at the end of a trimmed track.",
            },
            "format": {"type": "string", "enum": ["mp3", "wav"], "default": "mp3"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "is_instrumental", "lyrics", "duration_seconds"]
    side_effects = ["writes audio file(s) to output_path", "calls fal.ai API (paid)"]
    user_visible_verification = ["Listen to the track for mood fit, clean ending and no vocals when instrumental"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._get_api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return COST_PER_GENERATION_USD

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 90.0

    # ------------------------------------------------------------------

    @staticmethod
    def validate_inputs(inputs: dict[str, Any]) -> str | None:
        """Return an error message for inputs the API would reject, else None."""
        prompt = inputs.get("prompt") or ""
        if not (PROMPT_MIN_CHARS <= len(prompt) <= PROMPT_MAX_CHARS):
            return (
                f"prompt must be {PROMPT_MIN_CHARS}-{PROMPT_MAX_CHARS} characters "
                f"(got {len(prompt)})"
            )
        lyrics = inputs.get("lyrics") or ""
        if len(lyrics) > LYRICS_MAX_CHARS:
            return f"lyrics must be at most {LYRICS_MAX_CHARS} characters (got {len(lyrics)})"
        instrumental = inputs.get("is_instrumental", True)
        if not instrumental and not lyrics and not inputs.get("lyrics_optimizer", False):
            return "is_instrumental is false: provide lyrics or set lyrics_optimizer true"
        fmt = inputs.get("format", "mp3")
        if fmt not in ("mp3", "wav"):
            return f"format must be mp3 or wav (got {fmt!r})"
        duration = inputs.get("duration_seconds")
        if duration is not None and duration <= 0:
            return "duration_seconds must be positive"
        return None

    @staticmethod
    def build_payload(inputs: dict[str, Any]) -> dict[str, Any]:
        instrumental = bool(inputs.get("is_instrumental", True))
        payload: dict[str, Any] = {
            "prompt": inputs["prompt"],
            "is_instrumental": instrumental,
            # Integers, not strings: the API answers "44100" with 422 literal_error
            # even though the model page shows the defaults quoted.
            "audio_setting": {
                "sample_rate": 44100,
                "bitrate": 256000,
                "format": inputs.get("format", "mp3"),
            },
        }
        if not instrumental:
            if inputs.get("lyrics"):
                payload["lyrics"] = inputs["lyrics"]
            if inputs.get("lyrics_optimizer"):
                payload["lyrics_optimizer"] = True
        return payload

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=False,
        )
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return None

    @staticmethod
    def _trim_with_fade(src: Path, dst: Path, duration: float, fade_out: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH; cannot trim to duration_seconds")
        fade = max(0.0, min(fade_out, duration))
        afilter = f"afade=t=out:st={duration - fade:.3f}:d={fade:.3f}" if fade > 0 else "anull"
        codec = ["-c:a", "pcm_s16le"] if dst.suffix.lower() == ".wav" else ["-c:a", "libmp3lame", "-b:a", "256k"]
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(src), "-t", f"{duration:.3f}",
             "-af", afilter, *codec, str(dst)],
            check=True, capture_output=True,
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="FAL_KEY not set. " + self.install_instructions)
        problem = self.validate_inputs(inputs)
        if problem:
            return ToolResult(success=False, error=f"minimax_music: {problem}")

        import requests

        start = time.time()
        fmt = inputs.get("format", "mp3")
        output_path = Path(inputs.get("output_path") or f"minimax_music_output.{fmt}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
        charged = False

        try:
            submit = requests.post(
                f"https://queue.fal.run/{FAL_ENDPOINT}",
                headers=headers, json=self.build_payload(inputs), timeout=30,
            )
            submit.raise_for_status()
            queue = submit.json()
            status_url, response_url = queue["status_url"], queue["response_url"]

            deadline = time.time() + POLL_TIMEOUT_SEC
            while True:
                time.sleep(POLL_INTERVAL_SEC)
                st = requests.get(status_url, headers=headers, timeout=15)
                st.raise_for_status()
                status = st.json().get("status", "UNKNOWN")
                if status == "COMPLETED":
                    charged = True  # completed with output (422 validation failures are handled below)
                    break
                if status in ("FAILED", "CANCELLED", "ERROR"):
                    return ToolResult(success=False, error=f"MiniMax music generation {status.lower()}")
                if time.time() > deadline:
                    return ToolResult(
                        success=False,
                        error=f"MiniMax music generation still {status} after {POLL_TIMEOUT_SEC}s "
                              f"(request may still complete and be billed: {response_url})",
                    )

            result = requests.get(response_url, headers=headers, timeout=30)
            if result.status_code == 422:
                # Input validation failure: the queue reports COMPLETED, but nothing was generated.
                return ToolResult(
                    success=False,
                    error=f"MiniMax music request rejected by fal input validation (422): {result.text[:500]}",
                )
            result.raise_for_status()
            audio_url = result.json()["audio"]["url"]
            audio = requests.get(audio_url, timeout=120)
            audio.raise_for_status()

            target = inputs.get("duration_seconds")
            raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}") if target else output_path
            raw_path.write_bytes(audio.content)
            raw_duration = self._probe_duration(raw_path)

            notes: list[str] = []
            if target:
                if raw_duration is not None and raw_duration < target:
                    notes.append(
                        f"generated track is {raw_duration:.1f}s, shorter than requested {target}s; not padded"
                    )
                    shutil.copyfile(raw_path, output_path)
                else:
                    self._trim_with_fade(raw_path, output_path, float(target),
                                         float(inputs.get("fade_out_seconds", 2.0)))
        except Exception as e:
            billed = " (generation completed, so it was likely billed)" if charged else ""
            return ToolResult(success=False, error=f"MiniMax music generation failed{billed}: {e}")

        final_duration = self._probe_duration(output_path)
        artifacts = [str(output_path)]
        if inputs.get("duration_seconds") and raw_path != output_path:
            artifacts.append(str(raw_path))
        return ToolResult(
            success=True,
            data={
                "provider": "minimax",
                "gateway": "fal.ai",
                "model": FAL_ENDPOINT,
                "prompt": inputs["prompt"],
                "is_instrumental": bool(inputs.get("is_instrumental", True)),
                "output": str(output_path),
                "raw_output": str(raw_path),
                "raw_duration_seconds": raw_duration,
                "duration_seconds": final_duration,
                "format": fmt,
                "notes": notes,
            },
            artifacts=artifacts,
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=FAL_ENDPOINT,
        )
