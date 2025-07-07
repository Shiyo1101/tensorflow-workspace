# TensorFlow Workspace

このプロジェクトは、NVIDIA GPUを活用したTensorFlowの機械学習開発を、DockerとVS Code Dev Containerを用いて簡単に行うための環境です。

-----

## ✅ 前提条件

この開発環境を利用するには、お使いのPC（ホストOS）に以下のソフトウェアがインストールされている必要があります。詳細は次の「環境構築手順」セクションを参照してください。

1.  NVIDIA ドライバ
2.  Docker
3.  NVIDIA Container Toolkit (Linuxのみ)
4.  Visual Studio Code
5.  Dev Containers 拡張機能

-----

## 🛠️ 環境構築手順

### 1\. 前提ソフトウェアのインストールと確認

#### 1.1. NVIDIA ドライバ

  * **説明**: ホストOSがNVIDIA GPUを認識し、DockerコンテナがGPUを利用するために**必須のソフトウェア**です。
  * **インストール**: [NVIDIA公式サイト](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)からお使いのGPUに適合する最新のドライバをダウンロードし、インストールしてください。
  * **確認方法**: ターミナル（またはコマンドプロンプト）で以下のコマンドを実行し、GPUの情報が表示されることを確認します。
    ```bash
    nvidia-smi
    ```

> **CUDA Toolkitに関する注意**
> このプロジェクトでは、必要なCUDAライブラリはDockerイメージ内に含まれているため、通常**ホストOSに別途CUDA Toolkitをインストールする必要はありません。** 上記のNVIDIAドライバのインストールが最も重要です。もし、Docker以外の目的でホストOS上にCUDA Toolkitが必要な場合は、[NVIDIA CUDA Toolkitダウンロードページ](https://developer.nvidia.com/cuda-downloads)の指示に従ってインストールしてください。

#### 1.2. Docker

  * **説明**: コンテナを管理・実行するためのプラットフォームです。
  * **インストール**:
      * **Windows/Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop/)公式サイトからダウンロードし、インストールします。
      * **Linux (Ubuntu)**: Docker公式サイトの手順に従い、インストールします。
  * **確認方法**: ターミナルで以下のコマンドを実行し、バージョン情報が表示されることを確認します。
    ```bash
    docker --version
    ```

#### 1.3. (学内限定) Dockerのプロキシ設定

  * **説明**: 大学のネットワーク内からコンテナイメージのダウンロードなどを行う場合に必要です。
  * **設定方法**:
      * **Linux (Ubuntu)**: `~/.docker/config.json`にプロキシ情報を記述します。
      * **Windows / Mac (Docker Desktop)**: Settings \> Resources \> PROXIES からGUIで設定します。
      * (詳細な手順は省略。必要に応じて前の回答を参照してください。)

#### 1.4. NVIDIA Container Toolkit (Linuxのみ)

  * **説明**: Linux上でDockerがNVIDIA GPUを利用できるようにするためのツールキットです。
  * **インストール**: ターミナルで以下のコマンドを実行します。
    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```
  * **確認方法**: 以下のテストコマンドを実行し、エラーなくGPU情報が表示されれば成功です。
    ```bash
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
    ```

#### 1.5. Visual Studio Code & Dev Containers拡張機能

  * **説明**: 開発用エディタと、コンテナ内での開発を可能にするための拡張機能です。
  * **インストール**:
    1.  [Visual Studio Code](https://code.visualstudio.com/)をインストールします。
    2.  VS Codeを起動し、拡張機能タブから`ms-vscode-remote.remote-containers`を検索してインストールします。
  * **確認方法**: VS Codeのコマンドパレット(`Ctrl+Shift+P` or `Cmd+Shift+P`)で`Dev Containers:`と入力した際に、関連コマンドが表示されることを確認します。

-----

### 2\. プロジェクトの起動

1.  **VS Codeを起動**し、コマンドパレット (`Ctrl+Shift+P` or `Cmd+Shift+P`) を開きます。
2.  `Dev Containers: Reopen in Container` を検索し、選択します。
3.  自動的にDockerイメージのビルドとコンテナの起動が開始されます。初回は数分かかる場合があります。

ビルドが完了すると、VS Codeのウィンドウがリロードされ、コンテナ内の開発環境に接続された状態になります。

-----

### 3\. 動作確認

環境が正しく構築されたかを確認します。

1.  VS Codeでコンテナ内のターミナルを開きます (`Ctrl+@` or `Cmd+@`)。

2.  以下のコマンドを実行し、GPUがTensorFlowに認識されているか確認します。

    ```bash
    python src/test/gpu.py
    ```

**成功時の出力例:**

```
INFO:tensorflow:✅ 1個のGPUが利用可能です:
INFO:tensorflow:  GPU 0: /physical_device:GPU:0
...
INFO:tensorflow:TensorFlowの動作確認が完了しました。
```

上記のようにGPUが利用可能である旨のメッセージが表示されれば、セットアップは成功です。開発を始めましょう！