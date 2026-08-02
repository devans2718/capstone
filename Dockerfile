# Base pins torch + CUDA together; runtime (not -devel) keeps the image small.
# Tag to be finalized during Task 1 once WavJEPA + fairseq2 version
# compatibility is verified end-to-end — until then this is a working default.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System libs that pip can't provide: audio I/O and codec support.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 sox git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependency layer cached separately from source so code edits don't reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source is bind-mounted over /workspace in dev (compose.yaml); baking it in
# here is what makes the pushed image self-contained for cloud runs.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["bash"]
