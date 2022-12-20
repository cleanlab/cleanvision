# clean-vision
clean-vision automatically identifies issues in image datasets. This Data Centric AI package is designed as a first step for any image project to find strage occurences such as duplicate images, overexposed pictures and hot pixels in your data that could negatively impact model performance.

Adding clean-vision into your pipeline is as simple as running the code below:
```python

from clean_vision.imagelab import Imagelab

imagelab = Imagelab(dataset_path)

# clean-vision identifies all issues within your dataset, or ones speficied with the "issue_types=" param
imagelab.find_issues()

# clean-vision provides a neat report of the found issues
imagelab.report()
```

At the moment this package a work in progress and supports the following checks below:
|     | Issue Check                                              | Description                                                                                                                                                                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Light Images                                                                                        | Images that are too bright/washed out in the dataset                                   |
| 2   | Dark Images                                                                                         | Images that are irregularly dark                                |                                  |

Feel free to submit any found bugs or desired future checks as an [issue][issue].


## Quickstart

Example collection of images you can run this library on is located here:
https://drive.google.com/drive/folders/16wJPl8W643w7Tp2J05v3OMu8EpECkXpD?usp=share_link

After downloading these files, get started by trying to run: `examples/run.py`

## License

clean-vision is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

clean-vision is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See [GNU Affero General Public LICENSE](https://github.com/cleanlab/clean-vision/blob/main/LICENSE) for details.

[issue]: https://github.com/cleanlab/clean-vision/issues/new
