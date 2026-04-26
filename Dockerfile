FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglew2.2 \
    libglfw3 \
    libosmesa6-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/varstif_locomotion

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN sed -i 's/solver_iter/solver_niter/g' \
    /usr/local/lib/python3.12/dist-packages/gymnasium/envs/mujoco/mujoco_rendering.py

COPY . .

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ENTRYPOINT ["/bin/bash"]
