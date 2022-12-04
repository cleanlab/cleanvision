import unittest
import ImageDataset as ImageDatasetClass

class Test(unittest.TestCase): #trivial useless unittest
    imgset = ImageDatasetClass.ImageDataset((128,128))
    def test_0_thumbnail(self):
       self.assertIsNotNone(self.imgset.thumbnailsize)
        
        
  
if __name__ == '__main__':
    unittest.main()

#ideas for unittest:
# see if different filetypes of the same image give similar results. 
        
        
        