# GPU-enabled training/testing image for the Argoverse 2 diffusion-based
# motion-forecasting model.
#
# Match this tag to your host GPU driver's max supported CUDA version
# (check `nvidia-smi`). Uses -runtime (not -devel): torch and av2 both ship
# prebuilt wheels, so nothing gets compiled inside the container, and the
# smaller runtime image is enough.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS e2e

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# ---------- system packages ----------
# - python3/pip/dev: Ubuntu 24.04 ships Python 3.12
# - git/wget/curl: general tooling
# - libgl1/libglib2.0-0: headless matplotlib/opencv rendering
#   (needed by av2's visualize_scenario, easy to miss without it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade --ignore-installed pip

# ---------- Python deps (torch GPU, av2-api, s5cmd, training extras) ----------
COPY .docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Data (Argoverse 2 scenarios, checkpoints) is pulled at runtime with s5cmd
# into a mounted volume — never baked into the image.
WORKDIR /workspace

COPY .docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
