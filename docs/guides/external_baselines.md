# External 4K baseline runners

[`mapperatorinator.py`](../../scripts/baselines/mapperatorinator.py) and
[`mug_diffusion.py`](../../scripts/baselines/mug_diffusion.py) generate one
chart per process from audio. Both accept `--audio`, `--star`, `--arm`,
`--output` (a new `.osu` path), and `--seed` (a nonnegative integer).
They write a JSON receipt beside the chart with controls, wall time including
imports and model loading, and peak process RSS. On macOS RSS is reported in
bytes; it is not a measurement of all Metal allocations. Audio is referenced
by filename and is not copied. Put a copy beside the chart to play it.

These are external programs, with separate Python environments and weights.
Both setups use Python 3.10 and require `ffmpeg` on `PATH`.
They do not load R2, supply reference head rows or timing, start web servers,
or calculate achieved star ratings. Requested star values use each model's
own conditioning vocabulary. Run a legality check on every output before
using it in an evaluation; successful generation alone does not establish
legality or quality.

## Controls

| Runner | Arm | Conditioning |
| --- | --- | --- |
| Mapperatorinator | `natural` | Star and 4K; hold ratio and descriptors unspecified; CFG 1 |
| Mapperatorinator | `high-ln` | `hold_note_ratio=0.8`; CFG 1 |
| Mapperatorinator | `inverse` | `descriptors=["gimmick/LN inverse"]`; hold ratio unspecified; CFG 3 |
| Mug-Diffusion | `natural` | `sr=star`; LN ratio and style tags unspecified |
| Mug-Diffusion | `high-ln` | `sr=star`, `ln_ratio=0.8`, `ln=1` |

The natural arms leave LN conditioning unknown, rather than requesting zero
LNs. Mug accepts stars from 1 through 8, matching its conditioning bins.
It uses 100 DDIM steps, CFG 5, eta 0, batch size 1, and the web UI's
default grid snapping and 90 ms minijack processing. Its receipt records
postprocessing counts, ignored orphan held frames, and held lanes at the
array end. Its `.raw-hitobjects.txt` sidecar preserves objects before snapping
and minijack processing. These files have the same privacy restrictions as
the final chart.

Mapperatorinator uses the upstream V32 inference preset, fp32, sequential
generation and maximum batch size 1. Point it directly at the mania weight
directory so timing generation uses that model as well. It cannot download a
second model: Hugging Face access is disabled during inference. To use the
upstream CLI directly, the relevant overrides are `gamemode=3 keycount=4
difficulty=3.5 hold_note_ratio=0.8`, or `descriptors='["gimmick/LN inverse"]'
cfg_scale=3.0`; `audio_path`, `output_path` and `model_path` select the files.
The tokenizer's exact name includes `gimmick/`; the display label `LN inverse`
alone is silently skipped by the upstream conditioning code. The runner checks
the descriptor against the local tokenizer before generation.

## Mapperatorinator setup

Use [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)
at commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa` and the
[V32 mania files](https://huggingface.co/OliBomby/Mapperatorinator-v32/tree/74f22583400d259bf424819e11027c17933efe54/gamemode%3D3).
Download only `gamemode=3/{config.json,generation_config.json,model.safetensors,tokenizer.json}`.
The code and model card state MIT licensing. Do not substitute the mini model.
The weight revision used was `74f22583400d259bf424819e11027c17933efe54`;
`model.safetensors` SHA-256 is
`3d5d1c2a01ad462bcc99bafaaf44174bc47be099724574b17cd68016be5a863f`.

Create a separate Python 3.10 environment. The inference imports also reach
some training dependencies; the desktop GUI dependencies are unnecessary.
The following direct pins were used on macOS arm64:

```sh
uv venv --python 3.10 /path/to/envs/mapper
uv pip install --no-cache --python /path/to/envs/mapper/bin/python \
  torch==2.8.0 torchaudio==2.8.0 numpy==1.26.4 scipy==1.14.1 transformers==4.57.3 \
  accelerate==1.12.0 pydub==0.25.1 nnAudio==0.3.4 PyYAML==6.0.3 \
  hydra-core==1.3.2 einops==0.8.1 pandas==2.3.3 datasets==4.8.3 \
  wandb==0.24.2 rosu-pp-py==3.1.0 peft==0.18.1 matplotlib==3.10.8 \
  lightning==2.6.0 rich==14.2.0 huggingface-hub==0.36.2 \
  'slider @ git+https://github.com/OliBomby/slider.git@ea28c484a0a542d6e094e25aa02e1529ff2c8313'
export MAPPERATORINATOR_REPO=/path/to/Mapperatorinator
export MAPPERATORINATOR_MODEL='/path/to/weights/mapper/gamemode=3'
export MAPPERATORINATOR_DEVICE=cpu
```

The upstream config constructor also reads two Whisper **config JSONs**, even
though the V32 checkpoint contains its backbone config. Prefetch those small
files into `hf-cache` beside `gamemode=3`; no Whisper weights are needed:

```sh
/path/to/envs/mapper/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
cache = Path(os.environ['MAPPERATORINATOR_MODEL']).parent / 'hf-cache'
for repo, revision in (
    ('openai/whisper-small', '973afd24965f72e36ca33b3055d56a652f456b4d'),
    ('openai/whisper-base', 'e37978b90ca9030d5170a5c07aadb050351a65bb'),
):
    path = Path(hf_hub_download(repo, 'config.json', revision=revision, cache_dir=cache, token=False))
    # The upstream constructor requests the default branch in offline mode.
    refs = path.parents[2] / 'refs'
    refs.mkdir(exist_ok=True)
    (refs / 'main').write_text(revision)
PY
nice -n 10 /path/to/envs/mapper/bin/python scripts/baselines/mapperatorinator.py \
  --audio /path/to/audio.mp3 --star 3.5 --arm inverse \
  --output /private/results/inverse.osu --seed 0
```

CPU is the runner default. `MAPPERATORINATOR_DEVICE=auto` selects MPS when
available. With Torch 2.8.0, an attempt on the older code revision exhausted
the MPS allocator during the high-LN scroll-velocity pass on a 24 GiB Mac.
The current pin uses CPU for the smoke tests; MPS was not retested after the
code upgrade. Do not disable its memory watermark to bypass that failure.
All runners set CPU thread limits to two, Torch interop threads to one, and
process niceness to at least 10. The preset's CUDA-only fast decoder disables
itself on CPU.

Pin the code as well as the weights. The pinned revision's
[exporter](https://github.com/OliBomby/Mapperatorinator/blob/5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa/osuT5/osuT5/inference/postprocessor.py)
pairs holds per lane and skips groups without timestamps. Older revision
`4d2e5481d4394a67f67633340828eacf1d9caffa` keeps one pending head across all
lanes and loses valid holds that overlap across lanes. A two-hold
parser/exporter round trip reproduces that loss on the older revision and
preserves both holds on the pinned revision. A final chart can be legal even
when export lost notes.

For drop diagnostics, the local setup adds one warning to
`osuT5/osuT5/inference/postprocessor.py` when a `HOLD_NOTE_END` has no pending
head in its lane. The pinned upstream code already warns about overwritten
or trailing incomplete heads and groups without timestamps. The local diff
is kept with private results and changes no exported notes.

## Mug-Diffusion setup

Use [Keytoyze/Mug-Diffusion](https://github.com/Keytoyze/Mug-Diffusion) at
commit `c2903d4e3d7216d94ef512a9872bc77423c6e0ac`. Its code is MIT;
its README restricts **weights and generated charts to non-commercial use**.
Keep both private. The runners contain no weights, audio or generated charts.

The upstream README links a public Windows bundle on Google Drive. Its
`Content-Length` was 4,410,202,343 bytes. HTTP range reads allow ZIP-directory
inspection and extraction of only `MugDiffusion/models/ckpt/model.ckpt`
(1,839,231,053 bytes) and `model.yaml` (3,859 bytes). Check free space before
each download. The smoke setup reserves a 100 GB free-space floor and keeps
environments, weights and results below 6 GB. A full bundle and Windows
dependencies need not be retained.
The checkpoint SHA-256 is
`af6ab91337d0ef6b518367082ac3f849448c6daaa01fd987678fb25ea44ca184`;
the YAML SHA-256 is
`aa70281a92f57e3abfed75e78d2bd455ed206dccd660b0893c7318abc3b26901`.

```sh
uv venv --python 3.10 /path/to/envs/mug
uv pip install --no-cache --python /path/to/envs/mug/bin/python \
  torch==2.8.0 numpy==1.26.4 scipy==1.14.1 librosa==0.10.2.post1 \
  einops==0.8.1 omegaconf==2.3.0 pytorch-lightning==1.9.5 \
  torchmetrics==1.6.3 opt-einsum==3.4.0 eyed3==0.9.9 \
  scikit-learn==1.6.1 setuptools==80.9.0
export MUG_DIFFUSION_REPO=/path/to/Mug-Diffusion
export MUG_DIFFUSION_WEIGHTS=/private/weights/mug
nice -n 10 /path/to/envs/mug/bin/python scripts/baselines/mug_diffusion.py \
  --audio /path/to/audio.mp3 --star 3.5 --arm high-ln \
  --output /private/results/high-ln.osu --seed 0
```

Apply two local changes to `scripts/mapping.py` before using the runner:
in `load_model_from_config`, replace `model.cuda()` with `model.cpu()`, and
pass `weights_only=False` to `torch.load` for the original upstream Lightning
checkpoint. Use only the intended upstream checkpoint with this loader.
Save the diff beside private results; do not push it to the upstream project.
The UI is not imported. Neither MinaCalc nor Gradio is needed to embed the
star, LN ratio and LN tag.

The training window is 32,768 audio frames, or 190.218 s. The runner follows
`webui.py`: load the entire audio, increase latent length to the next multiple
of 32, and zero-pad audio to match. The older `scripts/mapping.py` main routine
instead truncates at the training window; the runner only imports its model
loader. Longer-song support in the code does not establish quality beyond
the training duration. Padding is retained, so check whether output events
extend beyond the original audio.
