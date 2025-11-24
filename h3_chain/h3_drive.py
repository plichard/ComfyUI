#!/usr/bin/env python3
"""Drive the MiniMax H3 chain workflow from Python.

Submits one clip at a time to a running ComfyUI over its HTTP API, doing the
extend-chain bookkeeping (clip slots, context latents, head trimming) that is
painful to maintain by hand in the node graph.

Models stay resident in the server between clips, which is the whole point --
on a 24 GB card the DiT alone is 20 GB, so a fresh process per clip would
re-stream it every time.

    ./h3_drive.py --project jungle --clips clips.json            # whole chain
    ./h3_drive.py --project jungle --clips clips.json --only 3   # re-roll one
    ./h3_drive.py --project jungle --clips clips.json --megapixels 0.6 --steps 6

Stdlib only. Talks to the server, so run it from the host or the container.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_WORKFLOW = Path(__file__).with_name("minimax_h3_chain_api.json")
DEFAULT_SERVER = "http://127.0.0.1:8188"
POLL_SECONDS = 1.5
MAX_REFS = 9  # ref2va's autogrow cap

# Which weights to load, per backend. ROCm (gfx1100) has no FP8/FP4 hardware --
# its own startup log lists float8/nvfp4 as *emulated* -- so it runs the INT8
# DiT and INT4 text encoder. Blackwell runs the formats the model actually ships
# in. Upstream publishes separate fl2va and ref2va weights, so the DiT follows
# the clip's mode; the local ROCm set only has fl2va, which is what it has always
# used for both. Until 2026-08-21 the graph had a single UNETLoader on fl2va
# feeding the sampler for every clip, so ref clips denoised with fl2va weights
# driving the ref2va conditioning path. It never errored -- the two DiTs are
# structurally identical (932 tensors, same keys, 259/259 LoRA modules present
# in both) -- it was just the wrong base.
MODELS = {
    "rocm": {
        "unet": {"fl": "minimaxH3INT8INT4_fl2vaINT8Pruned.safetensors",
                 "ref": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"},
        "clip": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "lora": {"fl": "minimax_h3_turbo_v4_step600.safetensors",
                 "ref": "minimax_h3_turbo_v4_step600.safetensors"},
    },
    "cuda": {
        "unet": {"fl": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                 "ref": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"},
        "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        # fl2v ships a 4-step and an 8-step turbo; ref2v ships only a 4-step.
        "lora": {"fl": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                 "fl8": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                 "ref": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"},
    },
}


def detect_backend(server: str) -> str:
    """'rocm' or 'cuda', from the server's own torch build.

    device.type is 'cuda' on ROCm too (torch reuses the namespace), so the only
    honest signal is the torch version string: 2.10.0+rocm7.2.4 vs 2.8.0+cu128.
    """
    try:
        stats = api(server, "/system_stats")
    except (ApiError, ServerGone, SystemExit):
        raise SystemExit(
            f"cannot reach {server} to detect the backend.\n"
            f"  pass --backend rocm|cuda to skip detection (needed for --dry-run "
            f"with no server running).")
    ver = ((stats or {}).get("system") or {}).get("pytorch_version", "")
    return "rocm" if "rocm" in ver.lower() else "cuda"


def pick_models(backend: str, clip: dict) -> dict:
    """Resolve the three mode-dependent weights for one clip."""
    m = MODELS[backend]
    mode = clip["mode"]
    lora = m["lora"].get(mode)
    if backend == "cuda" and mode == "fl" and clip["steps"] > 4:
        lora = m["lora"]["fl8"]      # the 8-step turbo exists only for fl2v
    return {"unet": m["unet"][mode], "clip": m["clip"], "lora": lora}
VAE_STRIDE = 16  # H3 video latent is width/16 x height/16

# Mirrors ResolutionSelector (comfy_extras/nodes_resolution.py:76) so the driver
# can predict the generation size without running the graph. Verified against
# the logs: 0.4 MP -> 576x736, 0.2 MP -> 384x544.
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def target_resolution(aspect: str, megapixels: float, multiple: int = 32):
    if aspect not in ASPECT_RATIOS:
        raise SystemExit(f"unknown aspect {aspect!r}; pick one of "
                         f"{sorted(ASPECT_RATIOS)}")
    w_ratio, h_ratio = ASPECT_RATIOS[aspect]
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    return (round(w_ratio * scale / multiple) * multiple,
            round(h_ratio * scale / multiple) * multiple)


def meminfo_mb(*keys):
    """Read fields from /proc/meminfo in MB. Returns None off Linux."""
    try:
        want, out = set(keys), {}
        with open("/proc/meminfo") as f:
            for line in f:
                field = line.split(":")[0]
                if field in want:
                    out[field] = int(line.split()[1]) // 1024
        return out if len(out) == len(want) else None
    except OSError:
        return None


def ram_snapshot():
    """(available MB, swap used MB), or None where /proc/meminfo is absent."""
    m = meminfo_mb("MemAvailable", "SwapTotal", "SwapFree")
    if m is None:
        return None
    return m["MemAvailable"], m["SwapTotal"] - m["SwapFree"]


def latent_resolution(path: Path):
    """Pixel size a saved chain latent was generated at.

    safetensors puts a JSON header up front, so the shape is readable without
    touching the tensor data. video is (B, C, T, H, W) in latent units.
    """
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        shape = header["video"]["shape"]
        return shape[-1] * VAE_STRIDE, shape[-2] * VAE_STRIDE
    except Exception:
        return None


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class ServerGone(Exception):
    """The server stopped answering -- crashed, killed, or shut down."""


class ApiError(Exception):
    def __init__(self, code, body):
        self.code, self.body = code, body
        super().__init__(f"HTTP {code}")

    def report(self):
        """ComfyUI returns validation failures as {error, node_errors}."""
        try:
            d = json.loads(self.body)
        except ValueError:
            return f"HTTP {self.code}: {self.body[:2000]}"
        lines = [f"HTTP {self.code}"]
        err = d.get("error") or {}
        if err:
            lines.append(f"  {err.get('type', '?')}: {err.get('message', '')}")
            if err.get("details"):
                lines.append(f"    {err['details']}")
        for node_id, ne in (d.get("node_errors") or {}).items():
            lines.append(f"  node '{node_id}' ({ne.get('class_type', '?')}):")
            for e in ne.get("errors", []):
                lines.append(f"    {e.get('message', '')}  {e.get('details', '')}")
        return "\n".join(lines)


def api(server: str, path: str, payload=None, method=None):
    url = server.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode(errors="replace")) from None
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        # A reset mid-poll usually means the server died rather than a network
        # blip: a GPU page fault or OOM-kill takes the process out from under us.
        raise ServerGone(f"{type(e).__name__}: {e}") from None


# --------------------------------------------------------------------------
# Clip list
# --------------------------------------------------------------------------

KNOWN_KEYS = {"prompt", "mode", "first_frame", "last_frame", "ref_images",
              "seed", "duration", "steps", "name", "ref_image_size", "lora_low_vram"}


def load_clips(path: Path, args) -> list[dict]:
    """Read the clip list.

    Either a bare JSON list, or {"defaults": {...}, "clips": [...]}. Each clip
    is a string (prompt only) or an object.

    Precedence: a per-clip key wins, then a flag actually typed on the command
    line, then the file's `defaults` block, then the built-in. A flag you took
    the trouble to type has to beat a default sitting in a file, or
    `--steps 8` silently does nothing against `"defaults": {"steps": 4}`.
    """
    def pick(item, key, cli, builtin):
        if key in item:
            return item[key]
        if cli is not None:            # argparse default is None when untyped
            return cli
        return defaults.get(key, builtin)

    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        defaults, items = raw.get("defaults", {}), raw.get("clips")
        if items is None:
            raise SystemExit(f"{path}: object form needs a 'clips' list")
    elif isinstance(raw, list):
        defaults, items = {}, raw
    else:
        raise SystemExit(f"{path}: expected a list or {{defaults, clips}}")

    clips = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise SystemExit(f"{path}: clip {i} is neither a string nor an object")
        merged = {**defaults, **item}

        unknown = set(merged) - KNOWN_KEYS
        if unknown:
            raise SystemExit(
                f"{path}: clip {i} has unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(KNOWN_KEYS)}")
        if "prompt" not in merged:
            raise SystemExit(f"{path}: clip {i} has no 'prompt'")

        mode = pick(item, "mode", args.mode, "fl")
        if mode not in ("fl", "ref"):
            raise SystemExit(f"{path}: clip {i} mode must be 'fl' or 'ref', got {mode!r}")

        refs = merged.get("ref_images") or []
        if isinstance(refs, str):
            refs = [refs]
        first = merged.get("first_frame")
        last = merged.get("last_frame")

        # Catch a mode/asset mismatch here rather than 90 seconds into a run.
        if mode == "ref" and not refs:
            raise SystemExit(f"{path}: clip {i} is mode 'ref' but has no 'ref_images'")
        if len(refs) > MAX_REFS:
            raise SystemExit(
                f"{path}: clip {i} has {len(refs)} ref_images; ref2va accepts {MAX_REFS}")
        if mode == "ref" and (first or last):
            print(f"  note: clip {i} is mode 'ref'; first_frame/last_frame are ignored",
                  file=sys.stderr)
        if mode == "fl" and refs:
            print(f"  note: clip {i} is mode 'fl'; ref_images are ignored", file=sys.stderr)

        ref_size = pick(item, "ref_image_size", args.ref_image_size, "match")
        if ref_size not in ("match", "max"):
            raise SystemExit(f"{path}: clip {i} ref_image_size must be 'match' "
                             f"or 'max', got {ref_size!r}")

        low_vram = pick(item, "lora_low_vram", args.lora_low_vram, False)
        if not isinstance(low_vram, bool):
            raise SystemExit(f"{path}: clip {i} lora_low_vram must be true/false, "
                             f"got {low_vram!r}")

        seed = merged.get("seed")
        if seed is None:
            seed = (args.seed + i - 1) if args.seed is not None else random.randrange(2**63)

        clips.append({
            "index": i,
            "name": merged.get("name", f"clip{i:03d}"),
            "prompt": merged["prompt"],
            "mode": mode,
            "first_frame": first,
            "last_frame": last,
            "ref_images": refs,
            "ref_image_size": ref_size,
            "lora_low_vram": low_vram,
            "seed": int(seed),
            "duration": float(pick(item, "duration", args.duration, 5.0)),
            "steps": int(pick(item, "steps", args.steps, 4)),
        })
    return clips


# --------------------------------------------------------------------------
# Workflow patching
# --------------------------------------------------------------------------

def setv(wf: dict, node: str, key: str, value):
    """Set one input, failing loudly if the workflow drifted from this script."""
    if node not in wf:
        raise SystemExit(f"workflow has no node '{node}' -- did the JSON change?")
    inputs = wf[node]["inputs"]
    if key not in inputs:
        raise SystemExit(f"node '{node}' has no input '{key}' (has: {sorted(inputs)})")
    if isinstance(inputs[key], list):
        raise SystemExit(f"refusing to overwrite linked input {node}.{key} with a literal")
    inputs[key] = value


def _loader(image, title):
    return {"class_type": "LoadImage", "_meta": {"title": title},
            "inputs": {"image": image}}


def _scaler(src, title):
    return {"class_type": "easy imageScaleDown", "_meta": {"title": title},
            "inputs": {"images": [src, 0], "width": ["resolution", 0],
                       "height": ["resolution", 1], "crop": "center"}}


def wire_images(wf: dict, clip: dict):
    """Attach only the optional images used by this clip."""
    refs = clip["ref_images"]
    first = clip["first_frame"]

    # Strip the template's optional image inputs and add back only those this
    # clip supplies. MiniMaxH3ImageToVideo supports prompt-only generation.
    wf["fl2va"]["inputs"].pop("first_frame", None)
    wf["fl2va"]["inputs"].pop("last_frame", None)
    for stale in ("load_first", "scale_first", "load_last", "scale_last"):
        wf.pop(stale, None)

    for key in [k for k in wf["ref2va"]["inputs"] if k.startswith("ref_images.ref_image_")]:
        del wf["ref2va"]["inputs"][key]
    for stale in [k for k in list(wf) if k.startswith("load_ref")]:
        del wf[stale]

    if clip["mode"] == "fl":
        if first:
            wf["load_first"] = _loader(first, "first_frame source")
            wf["scale_first"] = _scaler("load_first", "centre-crop first_frame")
            wf["fl2va"]["inputs"]["first_frame"] = ["scale_first", 0]
        if clip["last_frame"]:
            wf["load_last"] = _loader(clip["last_frame"], "last_frame source")
            wf["scale_last"] = _scaler("load_last", "centre-crop last_frame")
            wf["fl2va"]["inputs"]["last_frame"] = ["scale_last", 0]
    else:
        for n, path in enumerate(refs):
            node = f"load_ref{n}"
            # Refs go in unscaled: ref2va sizes them itself (to the generation
            # area for 'match', 2048 short edge for 'max'), so pre-shrinking
            # here would throw away detail it wants.
            wf[node] = _loader(path, f"ref_image_{n} (unscaled)")
            wf["ref2va"]["inputs"][f"ref_images.ref_image_{n}"] = [node, 0]


def build_prompt(base: dict, clip: dict, args) -> dict:
    """Patch a copy of the workflow for one clip of the chain."""
    wf = json.loads(json.dumps(base))
    n = clip["index"]

    chosen = pick_models(args.backend, clip)
    setv(wf, "load_unet", "unet_name", chosen["unet"])
    setv(wf, "load_clip", "clip_name", chosen["clip"])
    setv(wf, "turbo_lora", "lora_name", chosen["lora"])

    setv(wf, "prompt", "value", clip["prompt"])
    setv(wf, "noise", "noise_seed", clip["seed"])
    setv(wf, "duration", "value", clip["duration"])
    setv(wf, "sigmas", "steps", clip["steps"])

    setv(wf, "resolution", "aspect_ratio", args.aspect)
    setv(wf, "resolution", "megapixels", args.megapixels)
    setv(wf, "attention", "attention", args.attention)
    setv(wf, "ref2va", "ref_image_size", clip["ref_image_size"])
    setv(wf, "motion_context", "context_length", str(args.context_length))
    setv(wf, "motion_context", "audio_context_length", args.audio_context_length)

    # bypass (False) applies the LoRA at run time and is sharpest, but its peak
    # VRAM is what OOMs an extend clip at 0.4 MP -- the traceback lands in
    # bypass_forward. merge (True) folds it into the weights instead.
    setv(wf, "turbo_lora", "low_vram", clip["lora_low_vram"])
    setv(wf, "use_ref", "value", clip["mode"] == "ref")
    setv(wf, "extend", "value", n > 1)
    wire_images(wf, clip)

    # Chain slots. Save writes clip N's fixed slot (a re-roll overwrites its
    # own reject); load reads clip N-1's. clip_index must be > 0 for a fixed
    # slot -- 0 means "newest file", which is not retry-safe.
    # Everything for a project lives in output/<project>/: videos as
    # <name>.mp4, chain latents as clip_NNNNN.safetensors. SaveVideo creates a
    # counter name first; keep_video_fixed replaces it after a successful run.
    #
    # The loader gets the exact filename rather than the folder. _resolve_latent_path
    # returns immediately on os.path.isfile (nodes.py:848), so this skips the
    # directory scan and its clip_index matching altogether -- the driver already
    # knows which slot it wants, and naming it cannot pick up the wrong file.
    # clip_index is ignored in this mode; it is set correctly anyway so the value
    # still reads true.
    setv(wf, "save_latent", "filename_prefix", f"{args.project}/clip")
    setv(wf, "save_latent", "clip_index", n)
    setv(wf, "load_context", "latent_path", latent_rel(args.project, max(1, n - 1)))
    setv(wf, "load_context", "clip_index", max(0, n - 1))
    setv(wf, "save_video", "filename_prefix", f"{args.project}/{clip['name']}")

    # The trim converts frames to an audio duration with its own fps, so it
    # has to agree with the mux or the sound is cut by the wrong amount.
    setv(wf, "trim", "fps", args.fps)
    setv(wf, "create_video", "fps", args.fps)
    return wf


# --------------------------------------------------------------------------
# Chain manifest -- guards the one mistake the schema cannot catch
# --------------------------------------------------------------------------

def project_dir(args) -> Path:
    return Path(args.output_dir) / args.project


def latent_rel(project: str, index: int) -> str:
    """A clip's latent slot, relative to ComfyUI's output directory.

    Must match what the save node produces: filename_prefix "<project>/clip"
    splits to folder <output>/<project> + filename "clip", and clip_index N > 0
    writes "clip_%05d.safetensors" (nodes.py:948-955). Change one, change both.
    """
    return f"{project}/clip_{index:05d}.safetensors"


def latent_slot(args, index: int) -> Path:
    """The same file as seen from wherever the driver is running."""
    return Path(args.output_dir) / latent_rel(args.project, index)


def manifest_path(args) -> Path:
    return project_dir(args) / "chain.json"


def read_manifest(args) -> dict:
    p = manifest_path(args)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def write_manifest(args, clip: dict):
    p = manifest_path(args)
    p.parent.mkdir(parents=True, exist_ok=True)
    man = read_manifest(args)
    man["resolution"] = {"aspect_ratio": args.aspect, "megapixels": args.megapixels}
    man.setdefault("clips", {})[str(clip["index"])] = {
        "name": clip["name"], "seed": clip["seed"], "duration": clip["duration"],
        "steps": clip["steps"], "mode": clip["mode"], "prompt": clip["prompt"],
    }
    p.write_text(json.dumps(man, indent=2))


def check_models(args, clips: list[dict]):
    """Fail before submitting if a chosen weight file is not on the server.

    The loaders take a COMBO of what is actually on disk, so a wrong name comes
    back as a validation error naming the node but not the fix. Check it here
    and say what is available instead.
    """
    try:
        info = {n: api(args.server, f"/object_info/{n}")
                for n in ("UNETLoader", "CLIPLoader", "MiniMaxH3TurboLoRA")}
    except (ApiError, ServerGone):
        return  # not worth failing the run over a diagnostic
    fields = {"UNETLoader": ("unet_name", "unet"),
              "CLIPLoader": ("clip_name", "clip"),
              "MiniMaxH3TurboLoRA": ("lora_name", "lora")}
    missing = []
    for clip in clips:
        chosen = pick_models(args.backend, clip)
        for node, (field, key) in fields.items():
            spec = (info.get(node) or {}).get(node, {})
            opts = spec.get("input", {}).get("required", {}).get(field, [None])[0]
            if isinstance(opts, list) and chosen[key] not in opts:
                missing.append((key, chosen[key], opts))
    if missing:
        key, name, opts = missing[0]
        raise SystemExit(
            f"backend '{args.backend}' wants {key} '{name}', which this server "
            f"does not have.\n  available: {opts}\n"
            f"  pass --backend explicitly if auto-detection guessed wrong.")


def check_chain(args, clips: list[dict]):
    """A clip that extends must match the resolution its predecessor was built at.

    motion_context's context_latent has to be the same resolution as the clip
    being generated, so re-rolling clip 3 at a different --megapixels than the
    chain was built with fails inside the node. Catch it before the 90 seconds.
    """
    first = clips[0]["index"]
    if first == 1:
        return
    prev = first - 1
    latent = latent_slot(args, prev)
    if not latent.is_file():
        raise SystemExit(
            f"clip {first} extends clip {prev}, but {latent} does not exist.\n"
            f"generate clip {prev} first, or start from clip 1.")

    # Read the predecessor's actual geometry rather than trusting a manifest,
    # so chains built before this check existed are covered too.
    was = latent_resolution(latent)
    now = target_resolution(args.aspect, args.megapixels)
    if was and was != now and not args.force_resolution:
        raise SystemExit(
            f"resolution mismatch: clip {prev}'s latent is {was[0]}x{was[1]}, "
            f"but {args.megapixels} MP {args.aspect!r} gives {now[0]}x{now[1]}.\n"
            f"motion_context needs the context latent to match the clip being "
            f"generated, so this would fail inside the node. Re-run the whole "
            f"chain at the new resolution, or pass --force-resolution.")


# --------------------------------------------------------------------------
# Submit and wait
# --------------------------------------------------------------------------

def free_models(server: str, label: str = ""):
    """Ask the server to drop loaded models between clips.

    ComfyUI keeps a model resident after a prompt so the next one need not
    re-read it. Across a chain that accumulates: measured 17.2 GB RSS after a
    clip, back to 5.4 GB after this call. --cache-none does not cover it -- that
    governs node outputs, not the model cache. The cost is re-reading the
    weights on the next clip; the benefit is a chain that does not creep toward
    swap.
    """
    try:
        api(server, "/free", {"unload_models": True, "free_memory": True})
    except (ApiError, ServerGone):
        return
    snap = ram_snapshot()
    if snap:
        print(f"  freed models{' after ' + label if label else ''}; "
              f"{snap[0]} MB available")


def stop_server_job(server: str):
    """Ask the server to abandon the running prompt.

    Without this the GPU keeps working on a job nobody is waiting for.
    """
    try:
        api(server, "/interrupt", {})
    except (ApiError, ServerGone, SystemExit):
        pass


def run_clip(server: str, wf: dict, label: str, min_free_mb: int = 0) -> dict:
    try:
        resp = api(server, "/prompt", {"prompt": wf})
    except ApiError as e:
        print(f"\n{label}: rejected before execution\n{e.report()}", file=sys.stderr)
        raise SystemExit(1)
    except ServerGone as e:
        print(f"\n{label}: cannot reach ComfyUI ({e})\n"
              f"  is the container up?  docker ps --filter name=comfyui-rocm",
              file=sys.stderr)
        raise SystemExit(4)

    prompt_id = resp["prompt_id"]
    print(f"{label}: queued {prompt_id}", flush=True)

    started = time.monotonic()
    low_water = None
    blips = 0
    try:
        while True:
            time.sleep(POLL_SECONDS)

            # Watchdog. The models are bigger than RAM, so an over-cached or
            # over-large run drives the box into swap, where the GPU sits at 0%
            # busy and the job effectively never finishes. Failing loudly beats
            # thrashing for an hour.
            snap = ram_snapshot() if min_free_mb > 0 else None
            if snap is not None:
                avail, swap = snap
                low_water = avail if low_water is None else min(low_water, avail)
                if avail < min_free_mb:
                    print(f"\r{label}: RAM WATCHDOG TRIPPED" + " " * 20, file=sys.stderr)
                    print(f"  {avail} MB available < {min_free_mb} MB floor "
                          f"(swap in use: {swap} MB)", file=sys.stderr)
                    print(f"  stopping the server job; it would thrash rather "
                          f"than finish.", file=sys.stderr)
                    stop_server_job(server)
                    print(f"  the model set (~39 GB) is larger than RAM, so this "
                          f"is a capacity limit, not a leak. Options: restart the\n"
                          f"  container between chains, lower --megapixels, or if "
                          f"caching is on, drop it (--cache-none).", file=sys.stderr)
                    raise SystemExit(3)

            try:
                hist = api(server, f"/history/{prompt_id}")
                blips = 0
            except ServerGone as e:
                # One dropped poll can be a restart or a hiccup; several in a
                # row means it is not coming back.
                blips += 1
                if blips < 3:
                    continue
                print(f"\r{label}: SERVER GONE" + " " * 24, file=sys.stderr)
                print(f"  {e}", file=sys.stderr)
                print(f"  ComfyUI stopped answering after {int(time.monotonic()-started)}s. "
                      f"Check for a crash:", file=sys.stderr)
                print(f"    docker ps -a --filter name=comfyui-rocm     "
                      f"# 139 = SIGSEGV", file=sys.stderr)
                print(f"    docker logs --tail 40 comfyui-rocm 2>&1 | grep -i "
                      f"'memory access fault\\|out of memory\\|Fatal'", file=sys.stderr)
                raise SystemExit(4)
            if hist and prompt_id in hist:
                entry = finish(hist[prompt_id], label, time.monotonic() - started)
                if low_water is not None:
                    print(f"  RAM low-water: {low_water} MB available")
                return entry

            # \r only overwrites on a terminal; piped to a file or a log it
            # would emit one line per poll.
            if sys.stdout.isatty():
                mins, secs = divmod(int(time.monotonic() - started), 60)
                extra = f"  RAM {low_water} MB" if low_water is not None else ""
                print(f"\r  running {mins:d}m{secs:02d}s{extra}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  interrupting server job...", file=sys.stderr)
        stop_server_job(server)
        raise SystemExit(130)


def report_failure(messages: list):
    """Render an execution_error readably.

    ComfyUI packs the failing node's `current_inputs` into the message, which
    for an image or latent node means a fully formatted tensor -- thousands of
    lines that bury the actual exception. Print only what identifies the fault.
    """
    for entry in messages:
        if not (isinstance(entry, list) and len(entry) == 2):
            continue
        name, data = entry
        if name != "execution_error" or not isinstance(data, dict):
            continue
        print(f"  node '{data.get('node_id', '?')}' ({data.get('node_type', '?')}) "
              f"raised {data.get('exception_type', '?')}", file=sys.stderr)
        for line in str(data.get("exception_message", "")).splitlines()[:6]:
            print(f"    {line}", file=sys.stderr)
        for f in [f.rstrip() for f in data.get("traceback", [])][-3:]:
            print(f"    | {f.strip().splitlines()[0]}", file=sys.stderr)
        print(f"    ({len(data.get('executed') or [])} nodes had already run)",
              file=sys.stderr)
        return
    for m in messages:
        print(f"  {str(m)[:300]}", file=sys.stderr)


def finish(entry: dict, label: str, elapsed: float) -> dict:
    status = entry.get("status") or {}
    print(f"\r{label}: {status.get('status_str', '?')} in {elapsed:.0f}s" + " " * 12)
    if status.get("status_str") != "success":
        print(f"{label}: FAILED", file=sys.stderr)
        report_failure(status.get("messages", []))
        raise SystemExit(1)
    for f in walk_files(entry.get("outputs") or {}):
        print(f"  -> {f}")
    return entry


def walk_files(outputs: dict):
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for items in node_out.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    sub = item.get("subfolder") or ""
                    yield f"{sub}/{item['filename']}" if sub else item["filename"]


def keep_video_fixed(entry: dict, args, clip: dict):
    """Replace this clip's fixed video only after SaveVideo succeeds."""
    save_output = (entry.get("outputs") or {}).get("save_video", {})
    files = [Path(path) for path in walk_files({"save_video": save_output})
             if Path(path).suffix.lower() == ".mp4"]
    if len(files) != 1:
        raise SystemExit(f"save_video returned {len(files)} mp4 files; expected one")

    output_dir = Path(args.output_dir).resolve()
    source = (output_dir / files[0]).resolve()
    destination = (project_dir(args) / f"{clip['name']}.mp4").resolve()
    try:
        source.relative_to(output_dir)
        destination.relative_to(output_dir)
    except ValueError:
        raise SystemExit("refusing to move a video outside the output folder") from None
    if not source.is_file():
        raise SystemExit(f"saved video not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    print(f"  => {destination.relative_to(output_dir)} (fixed clip filename)")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Drive the MiniMax H3 extend chain over the ComfyUI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project", required=True,
                   help="names the latent chain folder and the video folder")
    p.add_argument("--clips", required=True, type=Path,
                   help="clip list: a JSON list, or {defaults, clips}")

    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="ComfyUI's output folder, as seen from here")

    p.add_argument("--duration", type=float, default=None,
                   help="seconds per clip; snapped to the 17k+5 frame grid")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; clip N uses seed+N-1. omit for random")
    p.add_argument("--aspect", default="3:4 (Portrait Standard)")
    p.add_argument("--megapixels", type=float, default=0.4)
    p.add_argument("--fps", type=float, default=24.0)

    p.add_argument("--mode", choices=("ref", "fl"), default=None,
                   help="default when a clip does not say")
    p.add_argument("--ref-image-size", choices=("match", "max"), default=None,
                   help="default when a clip does not say. 'max' uses a 2048 "
                        "short edge for identity fidelity but its ref tokens "
                        "ride every sampling step (measured ~3x slower); "
                        "'match' sizes refs to the generation area")
    p.add_argument("--attention", default="comfy kitchen attention",
                   choices=("comfy kitchen attention", "pytorch attention"))
    p.add_argument("--context-length", type=int, default=22, choices=(5, 22, 39, 56),
                   help="frames of the previous clip pinned, then trimmed off")
    p.add_argument("--audio-context-length", type=int, default=24)

    p.add_argument("--only", type=int, default=None, metavar="N",
                   help="run just clip N (re-roll); needs clip N-1's latent")
    p.add_argument("--from-clip", type=int, default=1, metavar="N",
                   help="resume a chain at clip N")
    p.add_argument("--backend", choices=("auto", "rocm", "cuda"), default="auto",
                   help="which weight set to load. 'auto' asks the server: a "
                        "torch build tagged +rocm gets the INT8/INT4 set, "
                        "anything else the fp8/nvfp4 set")
    p.add_argument("--lora-low-vram", action="store_true", default=None,
                   help="merge the turbo LoRA instead of bypassing it: much "
                        "lower peak VRAM, slightly softer on a quantized base. "
                        "Needed for extend clips at higher resolutions")
    p.add_argument("--no-free-between-clips", dest="free_between_clips",
                   action="store_false", default=True,
                   help="keep models loaded between clips. Faster on a chain "
                        "that never switches model, but RSS grows across a run")
    p.add_argument("--force-resolution", action="store_true",
                   help="skip the chain resolution check")
    p.add_argument("--min-free-ram", type=float, default=4.0, metavar="GB",
                   help="watchdog: abort the run if available RAM drops below "
                        "this. 0 disables. Exits 3 when it trips")
    p.add_argument("--dry-run", action="store_true",
                   help="print what each clip would do, submit nothing")

    args = p.parse_args()

    if not args.workflow.is_file():
        raise SystemExit(f"workflow not found: {args.workflow}")
    base = json.loads(args.workflow.read_text())
    clips = load_clips(args.clips, args)

    if args.only is not None:
        selected = [c for c in clips if c["index"] == args.only]
        if not selected:
            raise SystemExit(f"--only {args.only}: no such clip (list has {len(clips)})")
    else:
        selected = [c for c in clips if c["index"] >= args.from_clip]
        if not selected:
            raise SystemExit(f"--from-clip {args.from_clip}: nothing to do")

    if args.backend == "auto":
        args.backend = detect_backend(args.server)

    if not args.dry_run:
        check_chain(args, selected)
        check_models(args, selected)

    w, h = target_resolution(args.aspect, args.megapixels)
    print(f"project '{args.project}': {len(selected)} of {len(clips)} clips, "
          f"{w}x{h} ({args.megapixels} MP), backend={args.backend}")

    min_free_mb = int(args.min_free_ram * 1024)
    if min_free_mb > 0:
        snap = ram_snapshot()
        if snap is None:
            print("  RAM watchdog: disabled (/proc/meminfo unavailable)")
            min_free_mb = 0
        else:
            print(f"  RAM watchdog: abort below {min_free_mb} MB "
                  f"(now {snap[0]} MB available, swap {snap[1]} MB)")

    started = time.monotonic()
    for clip in selected:
        label = f"clip {clip['index']:03d}"
        wf = build_prompt(base, clip, args)
        if clip["mode"] == "ref":
            assets = f"refs={clip['ref_images']} ref_image_size={clip['ref_image_size']}"
        else:
            frames = []
            if clip["first_frame"]:
                frames.append(f"first={clip['first_frame']}")
            if clip["last_frame"]:
                frames.append(f"last={clip['last_frame']}")
            assets = " ".join(frames) or "prompt-only"
        head = clip["prompt"].strip().splitlines()
        print(f"\n{label} [{clip['name']}]  {clip['mode']}  seed={clip['seed']}  "
              f"{clip['duration']}s  {clip['steps']}st  "
              f"{'extend' if clip['index'] > 1 else 'first'}")
        print(f"  {assets}")
        print(f"  \"{head[0][:96] if head else ''}\"")

        if args.dry_run:
            continue
        entry = run_clip(args.server, wf, label, min_free_mb)
        keep_video_fixed(entry, args, clip)
        write_manifest(args, clip)
        if args.free_between_clips:
            free_models(args.server, label)

    if not args.dry_run:
        mins, secs = divmod(int(time.monotonic() - started), 60)
        print(f"\ndone: {len(selected)} clips in {mins}m{secs:02d}s")


if __name__ == "__main__":
    main()
