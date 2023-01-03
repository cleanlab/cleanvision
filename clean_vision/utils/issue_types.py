from enum import Enum


class IssueType(Enum):
    def __new__(cls, value, property, threshold):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.property = property
        obj.threshold = threshold
        return obj

    def set_hyperparameters(self, hyperparams) -> None:
        threshold = hyperparams.get("threshold", None)
        if threshold is None:
            return
        if not (0 < threshold < 1):
            raise ValueError("Threshold must lie between 0 and 1")            
        print(f"Setting threshold for {self.value} images issue to {threshold}")
        self.threshold = threshold

    DARK_IMAGES = ("Dark", "Brightness", 0.22)
    LIGHT_IMAGES = ("Light", "Brightness", 0.05)

    def __str__(self):
        return self.value
