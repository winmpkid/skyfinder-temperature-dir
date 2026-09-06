# Shared settings for the SkyFinder baseline/LDS experiments; this is not the
# complete official DIR training configuration.
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

MANIFEST_PATH = "data/manifest.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
