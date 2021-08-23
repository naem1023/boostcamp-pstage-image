from torch.utils.data import DataLoader


class MaskDataLoader(DatLoader):
    def __init__(self) -> None:
        super().__init__()

        # img, labels = next(iter(test_dataloader))
        # print(img)
        # print(labels)
