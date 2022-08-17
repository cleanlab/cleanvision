import unittest
from PIL import Image, ImageDraw, ImageFilter
from image_data_quality.image_dataset import ImageDataset
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
from image_data_quality.utils.utils import get_total_num_issues
imgset = ImageDataset(thumbnail_size = (128,128))
img1 = Image.new('RGB', (60, 30), color = (73, 109, 137))
img2 = img = Image.new('RGB', (100, 30), color = (102, 17, 51)) #"cleanlab"
img3 = Image.new('RGB', (60, 30), color = (73, 109, 137)) #duplicate of img3
img4 = Image.new('RGB', (100, 1), color = (0, 0, 0)) #all black
img5 = Image.new('RGB', (100, 30), color = (102, 17, 51)).filter(ImageFilter.GaussianBlur(radius = 5))
d = ImageDraw.Draw(img2)
d.text((10,10), "Cleanlab", fill=(234,151,17))
imgset.image_files = [img1, img2, img3, img4, img5]
sample = {"Blurriness":[2,5,3,9], "Brightness":[2,3], "Near Duplicates":[[1,17]]}
class Test(unittest.TestCase): #trivial useless unittest
    imgset = ImageDataset(thumbnail_size = (128,128))
    def test_0_thumbnail(self):
       self.assertIsNotNone(self.imgset.thumbnail_size)
    def test_1_check_brightness(self):
        self.assertEqual(check_brightness(img4), 0.0)
    def test_2_check_odd_size(self):
        self.assertTrue(check_odd_size(img4)<check_odd_size(img1))
    def test_3_check_entropy(self):
        self.assertTrue(check_entropy(img1)<check_entropy(img2))
    def test_4_check_static(self):
        self.assertTrue(check_static(img2)<check_static(img4))
    def test_5_check_blurriness(self):
        self.assertTrue(check_blurriness(img5)<check_blurriness(img2))
    def test_6_check_corrupt(self):
        self.assertTrue(check_corrupt(img2)==1)#not corrupted
    def test_7_get_total_num_issues(self):
        self.assertTrue(get_total_num_issues(sample)==8)

    
    
        
        
  
if __name__ == '__main__':
    unittest.main()

#ideas for unittest:
# see if different filetypes of the same image give similar results. 
        
        
        