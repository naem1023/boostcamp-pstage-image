test_dir = "/opt/ml/input/data/eval/images"
train_dir = "/opt/ml/input/data/train/images"

test_csv = "/opt/ml/input/data/eval/info.csv"
train_csv = "/opt/ml/input/data/train/train.csv"
with_system_path_csv = "/opt/ml/code/train-with-system-path.csv"

model_dir = "/opt/ml/code/saved_model"
BATCH_SIZE = 50

NUM_EPOCH = 6
k_split = 3
model_name = "efficientnet"

if model_name == "deit":
    LEARNING_RATE = 0.0005
else:
    LEARNING_RATE = 0.001

features = [
    "mask",
    "gender",
    "age",
]
