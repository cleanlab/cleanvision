import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import ImageStat, ImageFilter
from PIL.Image import Image

from cleanvision.issue_managers import IssueType


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
    def calculate(self, image: Image) -> Union[float, str]:
        raise NotImplementedError

    @abstractmethod
    def get_scores(self, **kwargs: Any) -> Any:
        self.check_params(**kwargs)
        return

    @staticmethod
    def mark_issue(
        scores: "np.ndarray[Any, Any]", threshold: float
    ) -> "np.ndarray[Any, Any]":
        return scores < threshold


def calc_avg_brightness(image: Image) -> Union[float, str]:
    stat = ImageStat.Stat(image)
    try:
        red, green, blue = stat.mean
    except ValueError:
        red, green, blue = (
            stat.mean[0],
            stat.mean[0],
            stat.mean[0],
        )  # deals with black and white images

    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255
    return cur_bright


def calc_percentile_brightness(
    image: Image, percentiles: List[int]
) -> Union[float, str]:
    imarr = np.asarray(image)
    if len(imarr.shape) == 3:
        r, g, b = imarr[:, :, 0], imarr[:, :, 1], imarr[:, :, 2]
        pixel_brightness = np.sqrt(0.241 * r * r + 0.691 * g * g + 0.068 * b * b)
    else:
        pixel_brightness = imarr
    return np.percentile(pixel_brightness, percentiles) / 255


class BrightnessProperty(ImageProperty):
    name: str = "brightness"

    def __init__(self, issue_type: str) -> None:
        self.issue_type = issue_type
        self.score_column = (
            "perc_99" if self.issue_type == IssueType.DARK.value else "perc_5"
        )

    def calculate(self, image: Image) -> Union[float, str, Dict]:
        percentiles = [1, 5, 10, 15, 90, 95, 99]
        perc_values = calc_percentile_brightness(image, percentiles=percentiles)
        raw_values = {f"perc_{p}": value for p, value in zip(percentiles, perc_values)}
        raw_values[self.name] = calc_avg_brightness(image)
        return raw_values

    def get_scores(
        self,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None  # all values are between 0 and 1
        if self.issue_type == IssueType.DARK.value:
            return raw_scores
        else:
            return 1 - raw_scores


def calc_aspect_ratio(image: Image) -> Union[float, str]:
    width, height = image.size
    size_score = min(width / height, height / width)  # consider extreme shapes
    assert isinstance(size_score, float)
    return size_score


class AspectRatioProperty(ImageProperty):
    name: str = "aspect_ratio"

    def __init__(self):
        self.score_column = self.name

    def calculate(self, image: Image) -> Union[float, str]:
        return {self.name: calc_aspect_ratio(image)}

    def get_scores(
        self,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None
        return raw_scores


def calc_entropy(image: Image) -> Union[float, str]:
    entropy = image.entropy()
    assert isinstance(
        entropy, float
    )  # PIL does not have type ann stub so need to assert function return
    return entropy


class EntropyProperty(ImageProperty):
    name: str = "entropy"

    def __init__(self):
        self.score_column = self.name

    def calculate(self, image: Image) -> Union[float, str]:
        return {self.name: calc_entropy(image)}

    def get_scores(
        self,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None

        scores: "np.ndarray[Any, Any]" = normalizing_factor * raw_scores
        scores[scores > 1] = 1
        return scores


def calc_blurriness(image: Image) -> Union[float, str]:
    edges = get_edges(image)
    blurriness = ImageStat.Stat(edges).var[0]
    assert isinstance(
        blurriness, float
    )  # ImageStat.Stat returns float but no typestub for package
    return blurriness


class BlurrinessProperty(ImageProperty):
    name = "blurriness"

    def __init__(self):
        self.score_column = self.name

    def calculate(self, image: Image) -> Union[float, str]:
        return {self.name: calc_blurriness(image)}

    def get_scores(
        self,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None
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


def calc_color_space(image: Image) -> Union[float, str]:
    return get_image_mode(image)


class ColorSpaceProperty(ImageProperty):
    name = "color_space"

    def __init__(self):
        self.score_column = self.name

    def calculate(self, image: Image) -> Union[float, str]:
        return {self.name: calc_color_space(image)}

    def get_scores(
        self,
        *,
        raw_scores: Optional[List[Union[float, str]]] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> "np.ndarray[Any, Any]":
        super().get_scores(**kwargs)
        assert raw_scores is not None
        scores = raw_scores.apply(lambda mode: 0 if mode == "L" else 1)
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
