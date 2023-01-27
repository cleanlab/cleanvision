import math
from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Any, Optional, Union

import numpy as np
from PIL import ImageStat, ImageFilter
from PIL.Image import Image

from cleanvision.issue_managers import IssueType

TImageProperty = TypeVar("TImageProperty", bound="ImageProperty")

TBrightnessProperty = TypeVar("TBrightnessProperty", bound="BrightnessProperty")
TAspectRatioProperty = TypeVar("TAspectRatioProperty", bound="AspectRatioProperty")
TEntropyProperty = TypeVar("TEntropyProperty", bound="EntropyProperty")
TBlurrinessProperty = TypeVar("TBlurrinessProperty", bound="BlurrinessProperty")
TColorSpaceProperty = TypeVar("TColorSpaceProperty", bound="ColorSpaceProperty")


class ImageProperty(ABC):
    @staticmethod
    def check_params(**kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "image": Image,
            "scores": "np.ndarray[Any, Any]",
            "threshold": float,
            "raw_scores": List[Union[float, str]],
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @abstractmethod
    def calculate(self: TImageProperty, image: Image) -> Union[float, str]:
        raise NotImplementedError

    @abstractmethod
    def get_scores(self: Any, **kwargs: Any) -> Any:
        self.check_params(**kwargs)
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

    def calculate(self: TBrightnessProperty, image: Image) -> Union[float, str]:
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
        self: Any,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        # reverse the brightness scores to catch images which are too bright
        if self.issue_type.name == IssueType.LIGHT.name:
            scores = 1 - scores
        return scores


class AspectRatioProperty(ImageProperty):
    name: str = "AspectRatio"

    def calculate(self: TAspectRatioProperty, image: Image) -> Union[float, str]:
        width, height = image.size
        size_score = min(width / height, height / width)  # consider extreme shapes
        assert isinstance(size_score, float)
        return size_score

    def get_scores(
        self: TAspectRatioProperty,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        return scores


class EntropyProperty(ImageProperty):
    name: str = "Entropy"

    def calculate(self: TEntropyProperty, image: Image) -> Union[float, str]:
        entropy = image.entropy()
        assert isinstance(
            entropy, float
        )  # PIL does not have type ann stub so need to assert function return
        return entropy

    def get_scores(
        self: TEntropyProperty,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        scores = np.array(raw_scores)
        scores: "np.ndarray[Any, Any]" = normalizing_factor * scores
        scores[scores > 1] = 1
        return scores


class BlurrinessProperty(ImageProperty):
    name = "blurriness"

    def calculate(self: TBlurrinessProperty, image: Image) -> Union[float, str]:
        edges = get_edges(image)
        blurriness = ImageStat.Stat(edges).var[0]
        assert isinstance(
            blurriness, float
        )  # ImageStat.Stat returns float but no typestub for package
        return blurriness

    def get_scores(
        self: TBlurrinessProperty,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        raw_scores = np.array(raw_scores)
        scores: "np.ndarray[Any, Any]" = 1 - np.exp(
            -1 * raw_scores * normalizing_factor
        )
        return scores


def get_edges(image: Image) -> Image:
    gray_image = image.convert("L")
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return edges


def calculate_brightness(red: float, green: float, blue: float) -> float:
    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255

    return cur_bright


class ColorSpaceProperty(ImageProperty):
    name = "color_space"

    def calculate(self: TColorSpaceProperty, image: Image) -> Union[float, str]:
        return get_image_mode(image)

    def get_scores(
        self: TColorSpaceProperty,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        scores = np.array([0 if mode is "L" else 1 for mode in raw_scores])
        return scores

    @staticmethod
    def mark_issue(
        scores: "np.ndarray[Any, Any]", threshold: float
    ) -> "np.ndarray[Any, Any]":
        issues: "np.ndarray[Any, Any]" = 1 - scores
        return issues.astype("bool")


def get_image_mode(image: Image) -> str:
    if image.mode:
        image_mode = image.mode
        assert isinstance(image_mode, str)
        return image_mode
    else:
        imarr = np.asarray(image)
        if len(imarr.shape) == 2 or (
            len(imarr.shape) == 3
            and (np.diff(imarr.reshape(-1, 3).T, axis=0) == 0).all()
        ):
            return "L"
        else:
            return "UNK"
