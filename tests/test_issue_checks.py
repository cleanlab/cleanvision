import pytest
from PIL import Image, ImageDraw, ImageFilter
from image_data_quality.issue_checks import (
    check_brightness,
    check_odd_size,
    check_entropy,
    check_static,
    check_blurriness,
    check_corrupt,
    check_duplicated,
    check_near_duplicates,
)
img1 = Image.new('RGB', (60, 30), color = (73, 109, 137))
img2 = Image.new('RGB', (100, 30), color = (102, 17, 51)) #"cleanlab"
img3 = Image.new('RGB', (60, 30), color = (73, 109, 137)) #duplicate of img3
img4 = Image.new('RGB', (100, 1), color = (0, 0, 0)) #all black
img5 = Image.new('RGB', (100, 30), color = (102, 17, 51)).filter(ImageFilter.GaussianBlur(radius = 5))
d = ImageDraw.Draw(img2)
d.text((10,10), "Cleanlab", fill=(234,151,17))
images = [img1, img2, img3, img4, img5]
def test_0_check_brightness():
    assert check_brightness(img4) == 0.0
def test_1_check_odd_size():
    assert check_odd_size(img4)<check_odd_size(img1)
def test_2_check_entropy():
    assert check_entropy(img1)<check_entropy(img2)
def test_3_check_static():
    assert check_static(img2)<check_static(img4)
def test_4_check_blurriness():
    assert check_blurriness(img5)<check_blurriness(img2)
def test_5_check_corrupt():
    assert check_corrupt(img2)==1 #not corrupted
def test_6_check_duplicated(issue_info = {}, misc_info={}):
    count = 1
    for img in images:
        check_duplicated(img, "img"+str(count), count, issue_info, misc_info)
        count+=1
    assert len(issue_info)==1 #only contains "Duplicated"
    assert issue_info["Duplicated"] == [3]
    assert len(misc_info)==3
    assert len(misc_info["Hash to Image"]) == 4 #5 images, 1 pair of duplicates
def test_7_check_near_duplicates(issue_info = {}, misc_info={}):
    count = 1
    for img in images:
        check_near_duplicates(img, "img"+str(count), count, issue_info, misc_info, **{"hash_size":10})
        count+=1
    assert len(issue_info["Near Duplicates"][0])==2 #only img1 and img3
    assert len(misc_info) == 3
    assert len(misc_info['Near Duplicate Imagehashes']) == 4 #img1 and img3 has the same hash