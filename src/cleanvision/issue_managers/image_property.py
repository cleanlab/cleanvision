from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, overload, Optional

import numpy as np
import pandas as pd
from PIL import ImageStat, ImageFilter
from PIL.Image import Image

from cleanvision.issue_managers import IssueType
from cleanvision.utils.constants import MAX_RESOLUTION_FOR_BLURRY_DETECTION
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname


class ImageProperty(ABC):
    name: str

    @property
    @abstractmethod
    def score_columns(self) -> List[str]:
        pass

    @staticmethod
    def check_params(**kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "image": Image,
            "dark_issue_data": pd.DataFrame,
            "threshold": float,
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if value is not None and not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @abstractmethod
    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        raise NotImplementedError

    @abstractmethod
    def get_scores(
        self, raw_scores: pd.DataFrame, issue_type: str, **kwargs: Any
    ) -> Any:
        self.check_params(**kwargs)
        return

    def mark_issue(
        self, scores: pd.DataFrame, threshold: float, issue_type: str
    ) -> pd.DataFrame:
        is_issue = pd.DataFrame(index=scores.index)
        is_issue[get_is_issue_colname(issue_type)] = (
            scores[get_score_colname(issue_type)] < threshold
        )
        return is_issue


def calc_avg_brightness(image: Image) -> float:
    stat = ImageStat.Stat(image)
    try:
        red, green, blue = stat.mean
    except ValueError:
        red, green, blue = (
            stat.mean[0],
            stat.mean[0],
            stat.mean[0],
        )  # deals with black and white images

    cur_bright: float = calculate_brightness(red, green, blue)
    return cur_bright


@overload
def calculate_brightness(red: float, green: float, blue: float) -> float:
    ...


@overload
def calculate_brightness(
    red: "np.ndarray[Any, Any]",
    green: "np.ndarray[Any, Any]",
    blue: "np.ndarray[Any, Any]",
) -> "np.ndarray[Any, Any]":
    ...


def calculate_brightness(
    red: Union[float, "np.ndarray[Any, Any]"],
    green: Union[float, "np.ndarray[Any, Any]"],
    blue: Union[float, "np.ndarray[Any, Any]"],
) -> Union[float, "np.ndarray[Any, Any]"]:
    cur_bright = (
        np.sqrt(0.241 * (red * red) + 0.691 * (green * green) + 0.068 * (blue * blue))
    ) / 255

    return cur_bright


def calc_percentile_brightness(
    image: Image, percentiles: List[int]
) -> "np.ndarray[Any, Any]":
    imarr = np.asarray(image)
    if len(imarr.shape) == 3:
        r, g, b = (
            imarr[:, :, 0].astype("int"),
            imarr[:, :, 1].astype("int"),
            imarr[:, :, 2].astype("int"),
        )
        pixel_brightness = calculate_brightness(
            r, g, b
        )  # np.sqrt(0.241 * r * r + 0.691 * g * g + 0.068 * b * b)
    else:
        pixel_brightness = imarr / 255.0
    perc_values: "np.ndarray[Any, Any]" = np.percentile(pixel_brightness, percentiles)
    return perc_values


class BrightnessProperty(ImageProperty):
    name: str = "brightness"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self, issue_type: str) -> None:
        self.issue_type = issue_type
        self._score_columns = [
            "brightness_perc_99"
            if self.issue_type == IssueType.DARK.value
            else "brightness_perc_5"
        ]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        percentiles = [1, 5, 10, 15, 90, 95, 99]
        perc_values = calc_percentile_brightness(image, percentiles=percentiles)
        raw_values = {
            f"brightness_perc_{p}": value for p, value in zip(percentiles, perc_values)
        }
        raw_values[self.name] = calc_avg_brightness(image)
        return raw_values

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None  # all values are between 0 and 1
        scores = pd.DataFrame(index=raw_scores.index)

        if issue_type == IssueType.DARK.value:
            scores[get_score_colname(issue_type)] = raw_scores[self.score_columns[0]]
        else:
            scores[get_score_colname(issue_type)] = (
                1 - raw_scores[self.score_columns[0]]
            )
        return scores


def calc_aspect_ratio(image: Image) -> float:
    width, height = image.size
    size_score = min(width / height, height / width)  # consider extreme shapes
    assert isinstance(size_score, float)
    return size_score


class AspectRatioProperty(ImageProperty):
    name: str = "aspect_ratio"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_aspect_ratio(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = raw_scores[self.score_columns[0]]
        return scores


def calc_entropy(image: Image) -> float:
    entropy = image.entropy()
    assert isinstance(
        entropy, float
    )  # PIL does not have type ann stub so need to assert function return
    return entropy


class EntropyProperty(ImageProperty):
    name: str = "entropy"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_entropy(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None
        scores = pd.DataFrame(index=raw_scores.index)
        scores_data = normalizing_factor * raw_scores[self.score_columns[0]]
        scores_data[scores_data > 1] = 1
        scores[get_score_colname(issue_type)] = scores_data
        return scores


def calc_blurriness(image: Image, max_resolution: int) -> float:
    ratio = max(image.width, image.height) / max_resolution
    if ratio > 1:
        low_rs = image.resize((int(image.width // ratio), int(image.height // ratio)))
    else:
        low_rs = image
    edges = get_edges(low_rs)
    blurriness = ImageStat.Stat(edges).var[0]
    assert isinstance(
        blurriness, float
    )  # ImageStat.Stat returns float but no typestub for package
    return np.sqrt(blurriness)  # type:ignore


class BlurrinessProperty(ImageProperty):
    name = "blurriness"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]
        self.max_resolution = MAX_RESOLUTION_FOR_BLURRY_DETECTION

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_blurriness(image, self.max_resolution)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        dark_issue_data: Optional[pd.DataFrame] = None,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert dark_issue_data is not None
        only_blur_scores = 1 - np.exp(-1 * raw_scores[self.name] * normalizing_factor)
        is_dark = dark_issue_data[get_is_issue_colname(IssueType.DARK.value)].astype(
            "int"
        )
        dark_score = dark_issue_data[get_score_colname(IssueType.DARK.value)]
        blur_scores = np.minimum(only_blur_scores + is_dark * (1 - dark_score), 1)
        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = blur_scores
        return scores


def get_edges(image: Image) -> Image:
    gray_image = image.convert("L")
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return edges


def calc_color_space(image: Image) -> str:
    return get_image_mode(image)


class ColorSpaceProperty(ImageProperty):
    name = "color_space"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_color_space(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None
        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = [
            0 if x == "L" else 1 for x in raw_scores[self.score_columns[0]]
        ]
        return scores

    def mark_issue(
        self, scores: pd.DataFrame, threshold: float, issue_type: str
    ) -> pd.DataFrame:
        is_issue = pd.DataFrame(index=scores.index)
        is_issue[get_is_issue_colname(issue_type)] = (
            1 - scores[get_score_colname(issue_type)]
        ).astype("bool")
        return is_issue


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
