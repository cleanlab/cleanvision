import math
from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Any, Type, Optional

import numpy as np
from PIL import ImageStat, Image

from cleanvision.issue_managers import IssueType

TImageProperty = TypeVar("TImageProperty", bound="ImageProperty")
TBrightnessProperty = TypeVar("TBrightnessProperty", bound="BrightnessProperty")
TAspectRatioProperty = TypeVar("TAspectRatioProperty", bound="AspectRatioProperty")
TEntropyProperty = TypeVar("TEntropyProperty", bound="EntropyProperty")


class ImageProperty(ABC):
    @staticmethod
    def check_params(*args: Any, **kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "image": Image,
            "scores": "np.ndarray[Any, Any]",
            "threshold": float,
            "raw_scores": Dict[Any, Any],
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @abstractmethod
    def calculate(self: TImageProperty, image: Image) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_scores(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.check_params(*args, **kwargs)
        return

    @staticmethod
    def mark_issue(
        scores: "np.ndarray[Any, Any]", threshold: float
    ) -> "np.ndarray[Any, Any]":
        return scores < threshold


class BrightnessProperty(ImageProperty):
    name: str = "Brightness"

    def __init__(self: TBrightnessProperty, issue_type: IssueType) -> None:
        self.issue_type = issue_type

    def calculate(self: TBrightnessProperty, image: Image) -> float:
        stat = ImageStat.Stat(image)
        try:
            red, green, blue = stat.mean
        except ValueError:
            red, green, blue = (
                stat.mean[0],
                stat.mean[0],
                stat.mean[0],
            )  # deals with black and white images

        cur_bright = calculate_brightness(red, green, blue)

        return cur_bright

    def get_scores(
        self: Any, raw_scores: Optional[List[float]] = None, *args: Any, **kwargs: Any
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(*args, **kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        # reverse the brightness scores to catch images which are too bright
        if self.issue_type.name == IssueType.LIGHT.name:
            scores = 1 - scores
        return scores


class AspectRatioProperty(ImageProperty):
    name: str = "AspectRatio"

    def calculate(self: TAspectRatioProperty, image: Image) -> float:
        width, height = image.size
        size_score = min(width / height, height / width)  # consider extreme shapes
        assert isinstance(size_score, float)
        return size_score

    def get_scores(
        self: TAspectRatioProperty,
        raw_scores: Optional[Dict[Any, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(*args, **kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        return scores


class EntropyProperty(ImageProperty):
    name: str = "Entropy"

    def calculate(self: TEntropyProperty, image: Image) -> float:
        entropy = image.entropy()
        assert isinstance(
            entropy, float
        )  # PIL does not have type ann stub so need to assert function return
        return entropy

    def get_scores(
        self: TEntropyProperty,
        raw_scores: Optional[Dict[Any, Any]] = None,
        normalizing_factor: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(*args, **kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        scores: "np.ndarray[Any, Any]" = normalizing_factor * scores
        scores[scores > 1] = 1
        return scores


def calculate_brightness(red: float, green: float, blue: float) -> float:
    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255

    return cur_bright
