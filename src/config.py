

# Model and tokenizer settings
MODEL_NAME = "Salesforce/codet5-small"
TOKENIZER_NAME = MODEL_NAME

# Dataset settings
DATASET_NAME = "mbpp"
DATASET_CONFIG = None  # Use full dataset, not sanitized
SPLIT = "train"  # Use training split for full dataset (~974 examples)

# Generation settings
PROMPT_PREFIX = "Generate unit tests for the following Python function:\n"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 8

# Training settings
OUTPUT_DIR = "./trained_model"
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-5