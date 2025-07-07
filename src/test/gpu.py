import logging
import sys

import tensorflow as tf

# ロギングの基本設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

logger.info("Python Version: %s", sys.version)
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        # 現在のGPUに対してメモリ成長を有効にする
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("✅ %d個のGPUが利用可能です:", len(gpus))
        for i, gpu in enumerate(gpus):
            logger.info("  GPU %d: %s", i, gpu.name)
    except RuntimeError as e:
        # メモリ成長はプログラムの初期化時に設定する必要がある
        logger.exception("GPUのメモリ成長設定中にエラーが発生しました: %s", e)
else:
    logger.warning("❌ GPUが見つかりませんでした。TensorFlowはCPUで実行されます。")

logger.info("-" * 30)
logger.info("簡単なテンソル演算を実行します...")
# 簡単な計算を実行してみる
a = tf.constant(10)
b = tf.constant(32)
c = a + b
logger.info("計算結果: %s + %s = %s", a, b, c.numpy())
logger.info("TensorFlowの動作確認が完了しました。")
