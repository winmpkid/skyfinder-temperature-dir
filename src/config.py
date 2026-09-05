# 本项目 SkyFinder baseline/LDS 的公共实验设置，不是 DIR 官方完整训练配置。
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

MANIFEST_PATH = "data/manifest.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
