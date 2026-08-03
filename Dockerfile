# Base pins torch + CUDA together; runtime (not -devel) keeps the image small.
# Tag finalized 2026-08-02: torch 2.7.0 + cu126 is exactly what WavJEPA upstream
# pins (torch==2.7.0, nvidia-*-cu12==12.6.*), so we inherit their tested combo
# instead of hoping ours is close enough. Was 2.5.1-cuda12.4 as a placeholder.
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# System libs that pip can't provide: audio I/O and codec support.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 sox git \
    && rm -rf /var/lib/apt/lists/*

# Checkpoints land on the /data mount, never in the image or a container layer:
# wavjepa-base alone is 785 MB, and /data/checkpoints is what restic tier 2 backs
# up. Same path locally and on cloud, so a warm cache is just a populated mount.
ENV HF_HOME=/data/checkpoints/hf

WORKDIR /workspace

# Dependency layer cached separately from source so code edits don't reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source is bind-mounted over /workspace in dev (compose.yaml); baking it in
# here is what makes the pushed image self-contained for cloud runs.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["bash"]
