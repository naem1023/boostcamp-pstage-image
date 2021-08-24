test_dir = "/opt/ml/input/data/eval/images"
train_dir = "/opt/ml/input/data/train/images"

test_csv = "/opt/ml/input/data/eval/info.csv"
train_csv = "/opt/ml/input/data/train/train.csv"
with_system_path_csv = "/opt/ml/code/train-with-system-path.csv"

model_dir = "/opt/ml/code/saved_model"
<<<<<<< HEAD
BATCH_SIZE = 150
LEARNING_RATE = 0.001
NUM_EPOCH = 10
k_split = 3
model_name = "BiT"

features = [
    "mask",
    "gender",
    "age",
]
=======
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCH = 5
k_split = 2
features = ["age", "mask", "gender"]
>>>>>>> 6852a782b6c0b56e054b91befbdaeffc962a878d
