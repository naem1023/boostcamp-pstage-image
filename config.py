test_dir = "D:\\dev\\train\\eval\\images"
train_dir = "D:\\dev\\train\\train\\images"

test_csv = "D:\\dev\\train\\eval\\info.csv"
train_csv = "D:\\dev\\train\\train\\train.csv"
with_system_path_csv = "D:\\dev\\boostcamp-pstage-image\\train-with-system-path.csv"

model_dir = ".\\saved_model"
BATCH_SIZE = 64

NUM_EPOCH = 20
k_split = 5
model_name = "efficientnet-b4"
ensemble = False
if model_name == "deit":
    LEARNING_RATE = 0.0005
else:
    LEARNING_RATE = 0.001

ray_tune = False
loss = "focal"
predict_dir = "2021-08-26T22:21:20.254632"
features = [
    "mask",
    "gender",
    "age",
]
