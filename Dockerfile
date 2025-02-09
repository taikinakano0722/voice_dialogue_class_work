# CUDA 12.2 ベースの PyTorch イメージを使用
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    wget git curl ffmpeg libsndfile1 libsox-fmt-mp3 \
    && rm -rf /var/lib/apt/lists/*

# シンボリックリンクの作成 (python3 → python)
RUN ln -s /usr/bin/python3.10 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# 仮想環境を作成
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip を最新にアップグレード
RUN pip install --upgrade pip

# Pythonライブラリのインストール
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install openai-whisper \
    && pip install transformers datasets \
    && pip install TTS \
    && pip install soundfile pydub scipy numpy \
    && pip install TTS[ja] \
    && pip install tensorboardX \
    && pip install fastchat \
    && pip install openai \
    && pip install accelerate \
    && pip install python-dotenv


# 作業ディレクトリを設定
WORKDIR /workspace

# コンテナ起動時に bash を開く
CMD ["/bin/bash"]
