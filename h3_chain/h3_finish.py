#!/usr/bin/env python3
"""Join an H3 chain and interpolate the finished video through ComfyUI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from h3_drive import DEFAULT_SERVER, run_clip


DEFAULT_MODEL = "film_net_fp16.safetensors"


def clip_names(project_dir: Path, clips_path: Path | None) -> list[str]:
    if clips_path is None:
        manifest_path = project_dir / "chain.json"
        if not manifest_path.is_file():
            raise SystemExit(f"chain manifest not found: {manifest_path}; pass --clips")
        raw = json.loads(manifest_path.read_text())
        clips = raw.get("clips") or {}
        try:
            names = [clips[key]["name"] for key in sorted(clips, key=int)]
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"invalid clip data in {manifest_path}") from None
    else:
        raw = json.loads(clips_path.read_text())
        items = raw.get("clips") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise SystemExit(f"{clips_path}: expected a list or {{defaults, clips}}")
        names = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                names.append(f"clip{index:03d}")
            elif isinstance(item, dict):
                names.append(item.get("name", f"clip{index:03d}"))
            else:
                raise SystemExit(f"{clips_path}: clip {index} is neither a string nor an object")

    if not names:
        raise SystemExit("the chain contains no clips")
    if any(not isinstance(name, str) or not name for name in names):
        raise SystemExit("every clip name must be a non-empty string")
    if len(names) != len(set(names)):
        raise SystemExit("clip names must be unique to select their latest takes")
    return names


def latest_take(project_dir: Path, name: str) -> Path:
    fixed = project_dir / f"{name}.mp4"
    if fixed.is_file():
        return fixed

    pattern = re.compile(rf"^{re.escape(name)}_(\d+)_\.mp4$")
    matches = []
    for path in project_dir.rglob("*.mp4"):
        match = pattern.fullmatch(path.relative_to(project_dir).as_posix())
        if match:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise SystemExit(f"no rendered take found for clip {name!r} in {project_dir}")
    return max(matches, key=lambda item: item[0])[1]


def concat_clips(ffmpeg: str, clips: list[Path], destination: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as concat_file:
        for clip in clips:
            escaped = str(clip.resolve()).replace("'", "'\\''")
            concat_file.write(f"file '{escaped}'\n")
        concat_file.flush()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file.name,
            "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
            "-movflags", "+faststart", str(destination),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        destination.unlink(missing_ok=True)
        detail = result.stderr.strip() or "ffmpeg exited without an error message"
        raise SystemExit(f"could not stitch the clips:\n{detail}")


def interpolation_workflow(source: str, project: str, output_name: str,
                           model: str, fps: float, multiplier: int) -> dict:
    return {
        "load_video": {
            "class_type": "VHS_LoadVideoPath",
            "_meta": {"title": "Load stitched H3 chain"},
            "inputs": {
                "video": source,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "load_interpolator": {
            "class_type": "FrameInterpolationModelLoader",
            "_meta": {"title": "FILM model"},
            "inputs": {"model_name": model},
        },
        "interpolate": {
            "class_type": "FrameInterpolate",
            "_meta": {"title": "Interpolate complete chain"},
            "inputs": {
                "interp_model": ["load_interpolator", 0],
                "images": ["load_video", 0],
                "multiplier": multiplier,
            },
        },
        "create_video": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Remux original chain audio"},
            "inputs": {
                "images": ["interpolate", 0],
                "audio": ["load_video", 2],
                "fps": fps * multiplier,
                "bit_depth": 8,
            },
        },
        "save_video": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save final video"},
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": f"{project}/{output_name}",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stitch the latest H3 clip takes and interpolate the final video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--clips", type=Path,
                        help="optional clip list; defaults to <project>/chain.json")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="ComfyUI output folder as seen by this command")
    parser.add_argument("--server-output-dir", type=Path, default=Path("/workspace/output"),
                        help="output folder as seen by the ComfyUI server")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model filename inside models/frame_interpolation")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="frame rate of the generated clips")
    parser.add_argument("--multiplier", type=int, choices=range(2, 17), default=2)
    parser.add_argument("--output-name", default="final_film",
                        help="final filename prefix inside the project folder")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")
    if args.fps * args.multiplier > 120:
        raise SystemExit("--fps * --multiplier must not exceed CreateVideo's 120 fps limit")
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise SystemExit(f"ffmpeg not found: {args.ffmpeg}")

    project_dir = args.output_dir / args.project
    if not project_dir.is_dir():
        raise SystemExit(f"project output folder not found: {project_dir}")
    names = clip_names(project_dir, args.clips)
    takes = [latest_take(project_dir, name) for name in names]

    print(f"project '{args.project}': {len(takes)} clips -> "
          f"{args.output_name}, FILM {args.multiplier}x, {args.fps:g} -> "
          f"{args.fps * args.multiplier:g} fps")
    for index, take in enumerate(takes, start=1):
        print(f"  {index:03d}: {take}")
    if args.dry_run:
        return

    project_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".h3_stitched_", suffix=".mp4",
                                     dir=project_dir, delete=False) as tmp:
        stitched = Path(tmp.name)
    try:
        print("\nstitching clips...", flush=True)
        concat_clips(ffmpeg, takes, stitched)

        server_source = args.server_output_dir / args.project / stitched.name
        workflow = interpolation_workflow(
            str(server_source), args.project, args.output_name,
            args.model, args.fps, args.multiplier)
        run_clip(args.server, workflow, "final interpolation")
    finally:
        stitched.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
