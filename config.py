test_dir = "/opt/ml/input/data/eval/images"
train_dir = "/opt/ml/input/data/train/images"

test_csv = "/opt/ml/input/data/eval/info.csv"
train_csv = "/opt/ml/input/data/train/train.csv"
with_system_path_csv = "/opt/ml/code/train-with-system-path.csv"

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCH = 5

features = ["age", "mask", "gender"]
