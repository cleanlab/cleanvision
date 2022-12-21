from enum import Enum


class IssueType(Enum):
    def __new__(cls, value, property, threshold):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.property = property
        obj._threshold = threshold
        return obj

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        if val is not None:
            if 0 < val < 1:
                print(f"Setting threshold for {self.value} images issue to {val}")
                self._threshold = val
            else:
                raise ValueError("Threshold must lie between 0 and 1")

    DARK_IMAGES = ("Dark", "Brightness", 0.22)
    LIGHT_IMAGES = ("Light", "Brightness", 0.05)

    def __str__(self):
        return self.value
