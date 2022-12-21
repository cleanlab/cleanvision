# clean-vision
clean-vision automatically identifies various issues in image datasets. This Data Centric AI package is designed as a quick first step for any computer vision project to find problems in your dataset (such as images which are: (near) duplicates, blurry, over/under-exposed, etc), which you may want to address before applying machine learning.

Adding clean-vision into your pipeline is as simple as running the code below:
```python

from clean_vision.imagelab import Imagelab

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(path)

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()  # add argument `issue_types` here to search for specific issues

# Produce a neat report of the issues found in your dataset
imagelab.report()
```

At the moment this package is a work in progress (expect sharp edges!) and supports the following checks:

|     | Issue Check                                              | Description                                                                                                                                                                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Light Images                                                                                        | Images that are too bright/washed out in the dataset                                   |
| 2   | Dark Images                                                                                         | Images that are irregularly dark                                |                                  |

Feel free to submit any found bugs or desired future checks as an [issue][issue].


## Quickstart

Example collection of images you can run this library on can be downloaded using:
```python
wget -nc 'https://cleanlab-public.s3.amazonaws.com/CleanVision/image_files.zip'
```
After downloading these files, get started by running: `python3 examples/run.py`


## Join our community

* The best place to learn is [our Slack community](https://cleanlab.ai/slack).  Join the discussion there to see how folks are using this library, discuss upcoming features, or ask for private support.

* Interested in contributing? See the [contributing guide](CONTRIBUTING.md). An easy starting point is to consider [issues](https://github.com/cleanlab/clean-vision/labels/good%20first%20issue) marked `good first issue` or simply reach out in [Slack](https://cleanlab.ai/slack). We welcome your help building a standard open-source library for data-centric computer vision!

* Ready to start adding your own code? See the [development guide](DEVELOPMENT.md).

* Have an issue? [Search existing issues](https://github.com/cleanlab/clean-vision/issues?q=is%3Aissue) or [submit a new issue](https://github.com/cleanlab/clean-vision/issues/new/choose).

* Have ideas for the future of data-centric computer vision? Check out [our active/planned Projects and what we could use your help with](https://github.com/cleanlab/clean-vision/projects).


## License

clean-vision is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

clean-vision is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See [GNU Affero General Public LICENSE](https://github.com/cleanlab/clean-vision/blob/main/LICENSE) for details.

[issue]: https://github.com/cleanlab/clean-vision/issues/new
