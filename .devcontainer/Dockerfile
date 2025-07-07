FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /tensorflow-workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# その他のPythonパッケージをインストール
RUN /bin/python -m pip install --no-cache-dir \
    numpy \
    opencv-python \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    ruff \
    pydantic 

# 環境変数の設定（CUDA関連）
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.2/lib64

# コンテナ起動時にシェルを開く設定
CMD ["bash"]