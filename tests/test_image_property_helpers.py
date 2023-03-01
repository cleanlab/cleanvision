import numpy as np
import pytest
from PIL import Image

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property import (
    BrightnessProperty,
    calculate_brightness,
    get_image_mode,
    calc_aspect_ratio,
    calc_entropy,
    calc_blurriness,
    get_edges,
    calc_color_space,
)


@pytest.mark.parametrize(
    "rgb,expected_brightness",
    [
        [(0, 0, 0), 0],
        [(255, 255, 255), 1],
        [(255, 0, 0), 0.49092],
        [(0, 255, 0), 0.83126],
        [(0, 0, 255), 0.26077],
    ],
    ids=["min", "max", "red", "green", "blue"],
)
def test_calculate_brightness(rgb, expected_brightness):
    brightness = calculate_brightness(*rgb)
    assert brightness == pytest.approx(expected=expected_brightness, abs=1e-5)


def test_calc_aspect_ratio():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    size_score = calc_aspect_ratio(img)  # min(width/height,height/width)
    assert size_score == 1


def test_calc_entropy():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    entropy_score = calc_entropy(img)  # min(width/height,height/width)
    assert entropy_score == img.entropy()


def test_calc_bluriness():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    edges = get_edges(img)
    blurriness = calc_blurriness(img)
    assert isinstance(edges, Image.Image)
    assert isinstance(blurriness, float)


def test_calc_color_space():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    img.mode = None
    color_space = calc_color_space(img)
    assert color_space == "UNK"


@pytest.mark.parametrize(
    "image,expected_mode",
    [
        [Image.new("RGB", (164, 164), (255, 255, 255)), "RGB"],
        [Image.new("RGB", (164, 164)), "RGB"],
        [Image.new("L", (164, 164)), "L"],
        [Image.new("RGB", (164, 164), (255, 160, 255)), "RGB"],
    ],
    ids=["white", "black", "grayscale", "rgb"],
)
def test_get_image_mode(image, expected_mode):
    mode = get_image_mode(image)
    assert mode == expected_mode


class TestBrightnessHelper:
    @pytest.fixture
    def image_property(self):
        return BrightnessProperty(IssueType.LIGHT)

    def test_init(self, image_property):
        assert isinstance(image_property, BrightnessProperty)
        assert hasattr(image_property, "issue_type")

    @pytest.mark.parametrize(
        "mock_mean,expected_output",
        [
            [[100], 0.39216],
            [[100, 200, 50], 0.68172],
        ],
        ids=["gray", "rgb"],
    )
    def test_calculate(self, image_property, monkeypatch, mock_mean, expected_output):
        from PIL import ImageStat

        class MockStat:
            def __init__(self, *args, **kwargs):
                pass

            @property
            def mean(self):
                return mock_mean

        monkeypatch.setattr(ImageStat, "Stat", MockStat)

        cur_bright = image_property.calculate("my_image")
        assert cur_bright == pytest.approx(expected=expected_output, abs=1e-5)

    def test_normalize(self, image_property, monkeypatch):
        raw_scores = [0.5, 0.3, 1.0, 1.2, 0.9, 0.1, 0.2]
        expected_output = np.array([0.5, 0.3, 1.0, 1.0, 0.9, 0.1, 0.2])

        with monkeypatch.context() as m:
            m.setattr(image_property, "issue_type", IssueType.DARK)
            normalized_scores = image_property.get_scores(raw_scores=raw_scores)
            assert all(normalized_scores == expected_output)

        normalized_scores = image_property.get_scores(raw_scores=raw_scores)
        assert all(normalized_scores == 1 - expected_output)

    @pytest.mark.parametrize(
        "scores,threshold,expected_mark",
        [
            [0.23, 0.4, [True]],
            [0.6, 0.5, [False]],
            [
                np.array([0.1, 0.2, 0.3, 0.4]),
                0.3,
                np.array([True, True, False, False]),
            ],
        ],
    )
    def test_mark_issue(self, image_property, scores, threshold, expected_mark):
        mark = image_property.mark_issue(np.array(scores), threshold)
        assert all(mark == expected_mark)
