import pytest
from PIL import Image

from cleanvision.utils.viz_manager import VizManager, truncate_titles


class TestVizManager:
    @pytest.mark.usefixtures("set_plt_show")
    @pytest.mark.parametrize(
        ("images", "title_info"),
        [
            ([Image.new("L", (100, 100))], {"name": ["image_title"]}),
            ([Image.new("L", (100, 100))] * 2, {"name": ["image_title"] * 4}),
            ([Image.new("L", (100, 100))] * 6, {"name": ["imaxge_title"] * 6}),
        ],
        ids=["plot single image", "plot <=4 images", "plt > 4 images"],
    )
    def test_individual_images(self, images, title_info):
        VizManager.individual_images(images, title_info, 4, (2, 2))

    @pytest.mark.usefixtures("set_plt_show")
    @pytest.mark.parametrize(
        ("image_sets", "title_info_sets"),
        [
            (
                [[Image.new("L", (100, 100))], [Image.new("L", (100, 100))] * 2],
                [{"name": ["image_title"]}, {"name": ["image_title"] * 2}],
            ),
        ],
    )
    def test_image_sets(self, image_sets, title_info_sets):
        VizManager.image_sets(image_sets, title_info_sets, 4, (2, 2))


def test_truncate_titles():
    assert truncate_titles(2, ["/home/usr/proj/dev/product/dataset/images/image_0001.img", "/home/usr/proj/dev"
                                                                                           "/product/dataset/images"
                                                                                           "/image_0002.img"]) == ["...es/image_0001.img", "...es/image_0002.img"]

    assert truncate_titles(2, ["image.jpeg", "image2.jpeg"]) == ["image.jpeg", "image2.jpeg"]
    assert truncate_titles(2, ["/pictures/mount/image_0001.img", "/home/usr/proj/dev"
                                                                 "/product/dataset/images"
                                                                 "/image_0002.img"]) == ["/pictures/mount/i...", "/home/usr/proj/de..."]
