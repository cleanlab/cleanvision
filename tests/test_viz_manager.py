import matplotlib.pyplot as plt
import pytest
from PIL import Image

from cleanvision.utils.viz_manager import VizManager


class TestVizManager:
    @pytest.fixture()
    def set_image_open(self, monkeypatch):
        def mock_open(path):
            return Image.new("L", (100, 100))

        monkeypatch.setattr(Image, "open", mock_open)

    @pytest.fixture()
    def set_plt_show(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)

    @pytest.mark.usefixtures("set_image_open", "set_plt_show")
    @pytest.mark.parametrize(
        ("filepaths", "titles"),
        [
            (["image.png"], ["image_title"]),
            (["image.png"] * 2, ["image_title"] * 2),
            (["image.png"] * 6, ["image_title"] * 6),
        ],
    )
    def test_individual_images(self, filepaths, titles, monkeypatch):
        VizManager.individual_images(filepaths, titles, 4, (2, 2))

    @pytest.mark.usefixtures("set_image_open", "set_plt_show")
    @pytest.mark.parametrize(
        ("filepath_sets", "title_sets"),
        [
            (
                [["image.png"], ["image.png"] * 2],
                [["image_title"], ["image_title"] * 2],
            ),
        ],
    )
    def test_image_sets(self, filepath_sets, title_sets):
        VizManager.image_sets(filepath_sets, title_sets, 4, (2, 2))
