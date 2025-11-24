# MiniMax H3 chain — API-format workflow

`minimax_h3_chain_api.json` is a driver-friendly copy of
`user/default/workflows/MiniMax_H3_00255_.json`, in the format `POST /prompt` expects.

Node IDs are semantic strings instead of numbers, so a driver mutates
`wf["prompt"]["inputs"]["value"]` rather than `wf["543"]["inputs"]["value"]`.
ComfyUI uses prompt keys verbatim (`execution.py:1130`), so arbitrary string IDs are fine.

Verified end to end — see *Status* at the bottom.

## Usage

```bash
./h3_drive.py --project jungle --clips clips.json                      # whole chain
./h3_drive.py --project jungle --clips clips.json --only 3             # re-roll clip 3
./h3_drive.py --project jungle --clips clips.json --megapixels 0.6 --steps 6
./h3_finish.py --project jungle                                       # join + FILM 2x
```

`h3_drive.py` is stdlib-only (no `websocket-client` in either environment, so it polls
`/history` rather than streaming progress). Run it from the host or the container — it
only talks to the server over HTTP.

### The clip list

A JSON list, or `{"defaults": {...}, "clips": [...]}` where `defaults` applies to every
clip. Precedence is **per-clip key > `defaults` > CLI flag**.

```json
{
  "defaults": { "duration": 5.0, "steps": 4 },
  "clips": [
    { "name": "establish", "mode": "fl",  "first_frame": "test.png",
      "prompt": "..." },
    { "name": "reveal",    "mode": "fl",  "first_frame": "a.png",
      "last_frame": "b.png", "prompt": "..." },
    { "name": "stand",     "mode": "ref", "ref_images": ["char.png", "outfit.png"],
      "duration": 4.0, "prompt": "..." }
  ]
}
```

| Key | Meaning |
|---|---|
| `prompt` | Required |
| `mode` | `fl` (prompt with optional keyframes) or `ref` (reference images) |
| `first_frame`, `last_frame` | Optional `fl` keyframes; omit both for prompt-only generation |
| `ref_images` | `ref` only; 1–9 paths, relative to ComfyUI's `input/` |
| `ref_image_size` | `ref` only; `match` (default) or `max` |
| `name` | Names the output video (default `clipNNN`) |
| `seed`, `duration`, `steps` | Per-clip overrides |

Unknown keys are rejected rather than silently ignored. An `fl` clip may omit both
keyframes and rely entirely on the prompt; a `ref` clip still requires `ref_images`.

Without `--seed`, each clip gets a random one; with `--seed N`, clip *i* uses `N+i-1`.
The seed used is always printed, so any clip can be reproduced exactly.

### Re-rolling one clip

`--only N` runs clip N alone, which is the loop for iterating on a prompt. It needs clip
N−1's latent to already exist, and the save node writes clip N's **fixed slot**, so each
re-roll overwrites its own previous attempt rather than stacking rejects or disturbing
the rest of the chain. The video also has a fixed name (`clip003.mp4`) and is replaced
only after a successful re-roll, so a failed attempt leaves the previous good file intact.

Before submitting, the driver reads clip N−1's latent header and compares its real pixel
size to what your flags would generate — see *Resolution must not change mid-chain*.

### Flags

| Flag | Why |
|---|---|
| `--dry-run` | Prints what each clip would do, submits nothing |
| `--only N` | Re-roll one clip |
| `--from-clip N` | Resume a chain at clip N |
| `--megapixels`, `--steps` | Raise quality for a full run |
| `--ref-image-size` | `match` (default) or `max`; per-clip key overrides it |
| `--attention` | `comfy kitchen attention` (default, ~42% faster) or `pytorch attention` |
| `--force-resolution` | Skip the chain resolution check |
| `--min-free-ram GB` | RAM watchdog, default 4; `0` disables |

Ctrl-C sends `/interrupt` so the GPU job actually stops rather than running on headless.

### Stitch and interpolate the final video

After the chain is rendered, `h3_finish.py` selects the fixed video for each
clip recorded in `output/<project>/chain.json`, stitches them in chain order, and runs
the complete result through `models/frame_interpolation/film_net_fp16.safetensors`.
The original joined audio is remuxed unchanged. FILM defaults to 2x, so a 24 fps chain
is saved at 48 fps with the same duration:

```bash
./h3_finish.py --project jungle
./h3_finish.py --project jungle --multiplier 4 --output-name final_film_4x
./h3_finish.py --project jungle --clips clips.json --dry-run
```

The temporary stitch uses ffmpeg stream copy, so it does not add a lossy encode before
interpolation, and is removed after the ComfyUI job. The server-side output path defaults
to the container mount `/workspace/output`. For a local server or a different mount,
override it, for example `--server-output-dir output`.

### RAM watchdog

The model set is ~39 GB against 60 GB of RAM, so a run that caches too much drives the
box into swap — where the GPU sits at **0% busy** and the job effectively never finishes.
That is much worse than failing: it looks alive.

The wait loop samples `/proc/meminfo` every poll. Below `--min-free-ram` (default 4 GB) it
posts `/interrupt`, explains that this is a capacity limit rather than a leak, and exits
**3** so a wrapper can tell it apart from an ordinary failure. Every successful clip also
reports its RAM low-water mark, so you can see how close a configuration runs.

It degrades safely: no `/proc/meminfo` (i.e. not Linux) prints `disabled` and carries on
rather than crashing.

### How images are wired

When supplied, `fl2va` resizes `first_frame` internally with crop `"disabled"`
(`comfy_extras/nodes_minimax_h3.py:132`) — it *stretches* to the target. So first/last
frames go through `easy imageScaleDown` with a centre crop first, or a square source
gets squashed into 3:4.

`ref2va` sizes references itself (`:218-231`) — to the generation area for `match`, or a
2048 short edge for `max`. **Reference images are therefore passed in unscaled**, unlike
the original UI workflow, which pre-shrank its ref to the generation size and threw away
the detail `max` exists to use.

Both branch nodes stay in every prompt even though only one executes. Their image inputs
are optional, so the driver removes the template image loaders and injects only the
first, last, or reference images used by the selected branch.

## Mutation points

Every driver-settable input is titled `KNOB ...` in `_meta.title`.

| Path | Meaning |
|---|---|
| `prompt.inputs.value` | Prompt text; feeds whichever branch is active |
| `noise.inputs.noise_seed` | Seed |
| `duration.inputs.value` | Clip length in **seconds** → snapped to the 17k+5 frame grid |
| `resolution.inputs.aspect_ratio` / `.megapixels` | Drives width/height everywhere |
| `use_ref.inputs.value` | `true` = REF2VA branch, `false` = FL2VA branch |
| `extend.inputs.value` | `false` for clip 1, `true` for clips 2..N |
| `load_first.inputs.image` | first_frame source, relative to `input/` |
| `load_ref0..N.inputs.image` | Reference images; `load_ref1+` are injected per clip |
| `load_context.inputs.latent_path` / `.clip_index` | Chain folder + **previous** clip |
| `save_latent.inputs.filename_prefix` / `.clip_index` | Chain folder + **this** clip |
| `save_video.inputs.filename_prefix` | Output video path |
| `sigmas.inputs.steps` | Sampler steps |
| `ref2va.inputs.ref_image_size` | `match` (default) or `max` — see below |
| `attention.inputs.attention` | `comfy kitchen attention` or `pytorch attention` |
| `motion_context.inputs.context_length` | One of `"5"`, `"22"`, `"39"`, `"56"` (string) |
| `trim.inputs.fps` + `create_video.inputs.fps` | **Must stay equal** — see Trim |

## Output layout

Everything for a project lives in one folder, `output/<project>/`:

```
output/<project>/
  chain.json               driver-written: resolution + per-clip provenance
  clip_00001.safetensors   chain latent, fixed slot per clip
  clip_00002.safetensors
  establish.mp4            video, fixed name from the clip's `name`
  turn_back.mp4
```

ComfyUI's SaveVideo node initially writes a counter filename. After a successful prompt,
the driver atomically replaces `<name>.mp4` with that result. Re-rolls therefore leave no
numbered versions, and a failed re-roll cannot destroy the previous good video. Latents
likewise use a fixed slot keyed on clip index.

The load node is pointed at the folder rather than a file and picks its slot by
`clip_index`. It only matches `*_%05d.safetensors` (`nodes.py:856`), so the videos and
`chain.json` sharing the folder are ignored.

## Chain bookkeeping

The save/load node pair has explicit clip-slot semantics
(`custom_nodes/ComfyUI-H3-Motion-Context/nodes.py:825`, `:935`). For clip N:

```
save_latent.filename_prefix = "<project>/clip"   # -> output/<project>/clip_0000N.safetensors
save_latent.clip_index      = N                  # must be > 0 for a fixed slot
load_context.latent_path    = "<project>"        # the folder, relative to output/
load_context.clip_index     = N - 1
extend.value                = (N > 1)
```

`clip_index > 0` writes a fixed slot, so re-rolling clip N overwrites its own reject
instead of stacking files. `clip_index = 0` means "newest file in the folder", which the
node's own docs call *not retry-safe* — avoid it in a loop.

Clip 1 submits fine even when the chain folder doesn't exist yet: `extend=false` makes the
lazy `easy ifElse` skip `load_context` entirely, and its `IS_CHANGED` swallows resolution
failures (`nodes.py:1012`).

## What changed from the UI workflow, and why

**Resolved away** (frontend-only indirection that has no meaning over the API):

- All KJNodes `SetNode`/`GetNode` and easy-use `setNode`/`getNode` pairs → direct links.
- The three duplicate `Size` subgraph instances → direct links from `resolution`.
- `SeedNode` ×2, `INTConstant` (steps), `StringConstant` (PROJECT_NAME) → plain widget values.

**Removed:**

- `UnloadAllModels` — it dropped every model after each run. That is actively wrong for a
  loop; keeping the DiT and text encoder warm between clips is the whole point of driving
  the server from Python.
- `PreviewImage` used as an image pass-through — saved a PNG per run for nothing. The
  scaled image now feeds the branches directly.
- `KSamplerSelect`, the `CLIP_NUMBER` `INTConstant`, and the `Clip Index` `SeedNode` — all
  three had no outgoing links.
- The `STEP1`–`STEP5` prompt bank — five string boxes you rewire by hand is exactly the
  bookkeeping a `for` loop replaces. One `prompt` node now.

**Fixed:**

- In the UI graph, `PROJECT_NAME` (`"test_project"`) was wired into *both* the save prefix
  and the load path, so the loader would have resolved the literal string `test_project`
  rather than a latent file. Those are now two independent knobs.
- `save_latent.filename_prefix` was `h3_context/redhead_jungle/` with a trailing slash,
  which splits to an empty filename and would have written `_00004.safetensors`. The
  existing file on disk is `redhead_jungle_00004.safetensors`, i.e. saved before that edit.
  The new prefix shape (`<folder>/clip`) is the one the loader can resolve by folder.
- Both branches took `length` from the same expression already, but their stale widgets
  disagreed (73 vs 124). Now there is one source.

**Preserved deliberately:** `context_length: "22"`, `audio_context_length: 24`,
`comfy kitchen attention`, `low_vram: false`, 4 steps, `simple` scheduler, 24 fps.

## ref_image_size

The driver defaults to **`match`**. The original UI workflow used `max`, and the cost of
that showed up plainly in the first full chain run:

```
clip 001  fl   124 frames   94s
clip 002  fl   124 frames   92s
clip 003  ref  107 frames  272s   <- ref_image_size: max
```

~3x the cost of an `fl` clip *despite being the shorter clip*, which is the node's own
warning made concrete: *"Reference tokens ride through every sampling step, so 'max' can
be several times slower."*

`max` uses a 2048 short edge for identity fidelity; `match` sizes references to the
generation area. Set it per clip, in `defaults`, or with `--ref-image-size max` for a
whole run — so `max` is now the opt-in for final renders rather than the price of every
iteration. The quality difference has not been measured here.

## Trim

`MiniMaxH3MotionContextTrim` now sits between the decoders and the mux, so delivered clips
no longer carry the pinned head and can be concatenated directly. It's the pack's own node
(`custom_nodes/ComfyUI-H3-Motion-Context/nodes.py:688`), which matters because it does two
things a plain image-slice node would not:

- **Cuts picture and sound by the same duration.** Trimming only the images would leave
  the audio `trim_frames` longer than the video and put the whole soundtrack ahead of the
  picture by `trim_frames/24` seconds.
- **`match_tail: true` pins each clip's audio to exactly `frames/fps`.** H3's audio latent
  runs at 40 Hz against 24 fps picture, so the grid rarely lands on a frame boundary and
  every clip ships ~8.3 ms too much or too little sound. That error *compounds* down a
  chain — 16.7 ms at the second seam, 25 ms at the third, growing without bound. This is
  worth leaving on even for clip 1, where `trim_frames` is 0.

`trim_frames` is wired from `motion_context` output 1 rather than hardcoded to
`context_length`, because the node returns what the encoder actually produced
(`nodes.py:685`), which isn't guaranteed to equal the requested window.

That wiring would normally force `motion_context` to run on clip 1 and fail, since a
dependency of an output node always executes. So it routes through `trim_count`, an
`easy ifElse` on `extend` that falls back to the `no_trim` constant (0). Laziness is
preserved: with `extend=false` the ifElse pulls `on_false` and `motion_context` never runs.

**`trim.inputs.fps` and `create_video.inputs.fps` must stay equal.** The trim converts
frames to an audio duration using its own `fps`; if the driver changes one and not the
other, the audio is cut by the wrong amount. Both are 24.0.

If you'd rather not carry the two extra nodes, the driver can set
`trim.inputs.trim_frames` to a literal (`22 if extend else 0`) and drop `trim_count` and
`no_trim` — slightly simpler, at the cost of assuming the encoder's count matches
`context_length`.

## Status

**Full 3-clip chain verified end to end** (2026-08-21), fresh container, `--cache-none`,
0.2 MP (384x544), seed 4242 — 3 clips in 7m37s, exit 0:

```
clip 001 [establish]  fl   success in  94s => chain2/establish.mp4
clip 002 [turn_back]  fl   success in  92s => chain2/turn_back.mp4
clip 003 [stand_up]   ref  success in 272s => chain2/stand_up.mp4
```

All three latent slots written, plus `chain.json`. Both extend clips did the real work:

```
loaded AV latent from .../chain2/clip_00002.safetensors
video from latent, 22 frames -> 7 cond blocks at 0..18, 107 frame clip at 384x544, trim 22
tail trimmed 267 samples (8.34ms) so audio matches 102 frames exactly
```

124 − 22 = 102, so `trim_count` pulled the encoder's real `trim_frames` through the
`extend` ifElse rather than the 0 constant. Clip 3's `duration: 4.0` snapped to 107
frames, distinct from the others' 124, and the head trim still applied on top. The
watchdog reported RAM never dipped below its floor (46 GB available at the end).

Covered: `fl` first clip, `fl` extend, `ref` extend, chain load, motion context, trim
(both `trim_frames=0` and `=22`), fixed-slot saves, manifest.

**Not yet exercised:** `last_frame`, more than one reference image, and the watchdog's
trip path (firing it would have interrupted a live run). All three are wired and lint
clean against `/object_info`, but have not been run.

Also checked statically: every class name, input name, output slot index and combo value
against `/object_info`, for `fl`-first-only, `fl`-first+last and `ref`x3 prompt shapes;
no cycles, no unreachable nodes; the dotted autogrow key `ref_images.ref_image_N` matches
how the executor flattens dynamic inputs (`comfy_api/latest/_io.py:1891`).

## VRAM accumulates across runs — restart when it OOMs

The first attempt died with `HIP out of memory ... 242.00 MiB is free`, while torch itself
accounted for only 3.64 GB. `POST /free {"unload_models":true,"free_memory":true}` freed
nothing. After `docker compose restart comfyui`:

```
before:  24225 MB used / 24560      free 0.24 GB
after:     256 MB used / 24560      free 23.80 GB
```

So roughly 20 GB was held by the ComfyUI process outside torch's allocator and outside
`drm-memory-vram` fdinfo accounting — not by the desktop, and not reclaimable without a
restart. It also visibly changes how much gets resident: `1162.76 MB usable` and
`367.74 MB loaded` of the DiT after the restart, against `131.42 MB usable` and
`0.00 MB loaded` before.

If a run OOMs, restart the container rather than hunting the workflow.

### Moving the desktop off the card did not help

With the desktop relocated to the iGPU (card2, 1458/2048 MB) the 7900 XTX starts a run
with the whole card free — and the DiT still gets almost none of it:

```
vram_free at idle                      23.81 GB
...text encoder load:   3329.55 MB usable,   0.00 MB loaded, 14257.82 MB offloaded
...video VAE load:      1708.27 MB usable, 1516.18 MB loaded
...DiT load:            1162.76 MB usable, 367.74 MB loaded, 19628.40 MB offloaded
```

~20 GB is gone **before the text encoder is even loaded**, and ComfyUI's own accounting
after a run shows `vram_free 2.48 GB` while `torch_vram_total` is only 1.86 GB — so
PyTorch is not the one holding it.

The strongest hypothesis, not yet proven: comfy-kitchen puts the INT8 DiT's ~19.6 GB of
quantized weights into VRAM through raw HIP during `UNETLoader`/`MiniMaxH3TurboLoRA`,
outside torch's allocator, while `model_management` still believes the weights live on CPU
and streams them anyway — the worst of both worlds, and consistent with `--cache-none`,
`/free` and extra headroom all making no difference.

Cheap way to confirm: poll `/system_stats` through a run and see whether `vram_free`
collapses at model-load time rather than during sampling. Worth testing
`MiniMaxH3TurboLoRA.low_vram = true` (its "merge" path) as a possible workaround.

One runtime footgun the schema can't catch: **resolution must not change mid-chain.**
`motion_context`'s `context_latent` tooltip says it "must be the same resolution as the
clip being generated", so resuming with `--from-clip 3 --megapixels 0.6` against a chain
built at 0.4 will fail on the first extended clip. Within a single invocation the driver
applies one resolution to every clip, so this only bites across separate runs.
