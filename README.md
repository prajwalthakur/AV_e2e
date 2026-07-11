# AV e2e — Diffusion-Based Motion Forecasting

Motion forecasting for autonomous vehicles using a diffusion-based model,
trained on [Argoverse 2](https://www.argoverse.org/av2.html) motion
forecasting scenarios via the [`av2`](https://github.com/argoverse/av2-api)
API.

## Status

Early scaffolding — the Docker/dev environment is set up; model and training
code land in `workspace/`.

## Environment

Everything runs out of a single GPU-enabled Docker image:

- **PyTorch (CUDA 12.6)** — installed from the official PyTorch wheel index,
  matched to the base image's CUDA runtime.
- **[`av2`](https://pypi.org/project/av2/)** — Argoverse 2 API, installed as
  a prebuilt wheel (no Rust toolchain needed).
- **[`s5cmd`](https://github.com/peak/s5cmd)** — fast, parallel S3 transfer
  for pulling datasets and checkpoints from AWS.
- `scipy`, `einops`, `tensorboard`, `tqdm`, `tyro`, `viser`, `dill` for
  training/eval and visualization.

The base image is `nvidia/cuda:12.6.3-runtime-ubuntu24.04` (not `-devel`) —
torch and av2 both ship prebuilt wheels, so nothing compiles inside the
container and the image stays small. If your GPU driver supports a
different CUDA version, update the tag in
[.docker/e2e.Dockerfile](.docker/e2e.Dockerfile) and the
`--extra-index-url` in [.docker/requirements.txt](.docker/requirements.txt)
to match (check `nvidia-smi` for your driver's max supported CUDA version).

Training runs on remote GPU servers; the same image runs locally for
small-scale testing on whatever GPU (or CPU) you have.

## Quickstart

Build the image:

```bash
./.scripts/.build/.build.sh
```

Start a dev container (repo root is bind-mounted to `/workspace`,
GPU attached by default):

```bash
./.scripts/.deploy/devel.sh        # GPU
./.scripts/.deploy/devel.sh -c     # CPU-only
```

Inside the container, verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Pulling data from S3

Use `s5cmd` (already installed in the image) rather than baking data into
the image — mount a host directory and sync into it, e.g.:

```bash
s5cmd cp 's3://<bucket>/av2/motion-forecasting/*' /workspace/data/av2/
```

AWS credentials: `devel.sh`/`base.sh` mount `~/.aws` into the container
read-only if it exists, and pass through `AWS_PROFILE`/`AWS_REGION` from
your host environment.

## Repo layout

```
.docker/            Dockerfile, entrypoint, Python requirements
.scripts/.build/    image build script
.scripts/.deploy/   container run/stop scripts (devel.sh is the main entrypoint)
.scripts/.setup/    host-side setup helpers (CUDA env vars, etc.)
workspace/           project code (data, models, training/eval scripts)
```

## License

MIT — see [LICENSE](LICENSE).
