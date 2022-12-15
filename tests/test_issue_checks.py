from PIL import Image
from clean_vision.issue_checks import check_grayscale


def test_grayscale():
    white_im = Image.new("RGB", (164, 164), (255, 255, 255))
    black_im = Image.new("RGB", (164, 164))
    greyscale_im = Image.new("L", (164, 164))
    color_im = Image.new("RGB", (164, 164), (255, 160, 255))
    assert check_grayscale(white_im) == 1
    assert check_grayscale(black_im) == 1
    assert check_grayscale(greyscale_im) == 1
    assert check_grayscale(color_im) == 0
