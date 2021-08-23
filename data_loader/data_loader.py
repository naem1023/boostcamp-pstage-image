transformation = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(224)
    ])
train_dataset = MaskDataset(train_dir_glob, transformation)
test_dataset = MaskDataset(test_dir_glob, transformation, train=False)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
img, labels = next(iter(train_dataloader))
# img, labels = next(iter(test_dataloader))
# print(img)
# print(labels)