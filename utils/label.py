"""
mask
    wear: 0
    incorrect: 1
    not wear: 2
gender
    male: 0
    female: 1
age
    <30: 0
    >=30 and <60: 1
    >=60: 2
"""


class FileNameError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class Label:
    def __init__(self):
        self.mask = [0, 1, 2]
        self.gender = [0, 1]
        self.age = [0, 1, 2]
        self.feature_func = {
            "gender": self.gender_feature,
            "mask": self.mask_feature,
            "age": self.age_feature,
        }

    def get_classes(self, feature) -> list:
        return getattr(self, feature)

    def mask_feature(self, path) -> int:
        file_name = path.split("/")[-1]
        if file_name[:4] == "mask":
            return 0
        elif file_name[:14] == "incorrect_mask":
            return 1
        elif file_name[:6] == "normal":
            return 2
        else:
            raise FileNameError("Mask naming error")

    def gender_feature(self, path) -> int:
        gender = path.split("/")[-2].split("_")[1]
        if gender == "male":
            return 0
        elif gender == "female":
            return 1
        else:
            raise FileNameError("Gender naming error")

    def age_feature(self, path) -> int:
        age = int(path.split("/")[-2][-2:])
        if age < 30:
            return 0
        elif 30 <= age < 60:
            return 1
        elif age >= 60:
            return 2
        else:
            raise FileNameError("Age naming error")

    def get_label(self, path: str, feature: str) -> int:
        try:
            return self.feature_func[feature](path)
        except FileNameError as e:
            print(e)
            exit()

