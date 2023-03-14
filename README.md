<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanvision_logo_open_source_transparent.png" width=50% height=50%>
</p>

<img width="1200" alt="Screen Shot 2023-03-10 at 10 23 33 AM" src="https://user-images.githubusercontent.com/10901697/224394144-bb0e1c85-6851-4828-bcd2-4ed234270a78.png">

CleanVision automatically detects potential issues in image datasets like images that are: blurry, under/over-exposed, (near) duplicates, etc.
This data-centric AI package is a quick first step for any computer vision project to find problems in the dataset, which you want to address before applying machine learning.
CleanVision is super simple -- run the same couple lines of Python code to audit any image dataset!

[![Read the Docs](https://readthedocs.org/projects/cleanvision/badge/?version=latest)](https://cleanvision.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/github/cleanlab/cleanvision/branch/main/graph/badge.svg?token=y1N6MluN9H)](https://codecov.io/gh/cleanlab/cleanvision)

## Installation

```shell
pip install git+https://github.com/cleanlab/cleanvision.git
```

## Quickstart

Download an example dataset (optional). Or just use any collection of image files you have.

```shell
wget -nc 'https://cleanlab-public.s3.amazonaws.com/CleanVision/image_files.zip'
```

Run CleanVision to audit the images.

```python
from cleanvision.imagelab import Imagelab

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab("path_to_dataset")

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()

# Produce a neat report of the issues found in your dataset
imagelab.report()
```

CleanVision diagnoses many types of issues, but you can also check for only specific issues.

```python
issue_types = {"dark": {}, "blurry": {}}

imagelab.find_issues(issue_types=issue_types)

# Produce a report with only the specified issue_types
imagelab.report(issue_types=issue_types)
```

## More resources on how to use CleanVision

- [Tutorial notebook](https://github.com/cleanlab/cleanvision/blob/main/examples/demo.ipynb)
- [Example script](https://github.com/cleanlab/cleanvision/blob/main/examples/run.py)
- [Additional example notebooks](https://github.com/cleanlab/cleanvision-examples)
- [Documentation](https://cleanvision.readthedocs.io/)

## *Clean* your data for better Computer *Vision*

The quality of machine learning models hinges on the quality of the data used to train them, but it is hard to manually identify all of the low-quality data in a big dataset. CleanVision helps you automatically identify common types of data issues lurking in image datasets.

This package currently detects issues in the raw images themselves, making it a useful tool for any computer vision
task such as: classification, segmentation, object detection, pose estimation, keypoint detection, [generative modeling](https://openai.com/research/dall-e-2-pre-training-mitigations), etc.
To detect issues in the labels of your image data, you can instead
use the [cleanlab](https://github.com/cleanlab/cleanlab/) package.

In any collection of image files (most [formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) supported), CleanVision can detect the following types of issues:

|     | Issue Type       | Description                                               | Issue Key        | Example                                                                                                                         |
|-----|------------------|-----------------------------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 1   | Dark             | Irregularly dark images                                   | dark             | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/dark.jpg)                         |
| 2   | Blurry           | Blurry or out of focus images                             | blurry           | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/blurry.png)           |
| 3   | Grayscale        | Images lacking color                                      | grayscale        | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/grayscale.jpg)        |
| 4   | Low Information  | Images lacking much information (e.g. stick figure image) | low_information  | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/low_information.jpg)  |
| 5   | Odd Aspect Ratio | Unusual aspect ratio (i.e. overly skinny/wide)            | odd_aspect_ratio | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/odd_aspect_ratio.jpg) |
| 6   | Light            | Too bright or mostly white images                         | light            | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/light.jpg)            |
| 7   | Exact Duplicates | Images that are exact duplicates of each other            | exact_duplicates | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/exact_duplicates.png) |
| 8   | Near Duplicates  | Images that are visually identical to each other          | near_duplicates  | ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/near_duplicates.png)  |


This package is still a work in progress, so expect sharp edges.
Feel free to submit any found bugs or desired functionality as an [issue][issue]!

## Join our community

* The best place to learn is [our Slack community](https://cleanlab.ai/slack). Join the discussion there to see how
  folks are using this library, discuss upcoming features, or ask for private support.

* Interested in contributing? See the [contributing guide](CONTRIBUTING.md). An easy starting point is to
  consider [issues](https://github.com/cleanlab/cleanvision/labels/good%20first%20issue) marked `good first issue` or
  simply reach out in [Slack](https://cleanlab.ai/slack). We welcome your help building a standard open-source library
  for data-centric computer vision!

* Ready to start adding your own code? See the [development guide](DEVELOPMENT.md).

* Have an issue? [Search existing issues](https://github.com/cleanlab/cleanvision/issues?q=is%3Aissue)
  or [submit a new issue](https://github.com/cleanlab/cleanvision/issues/new/choose).

* Have ideas for the future of data-centric computer vision? Check
  out [our active/planned Projects and what we could use your help with](https://github.com/cleanlab/cleanvision/projects).

## License

Copyright (c) 2017-2023 Cleanlab Inc.

cleanvision is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

cleanvision is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See [GNU Affero General Public LICENSE](https://github.com/cleanlab/cleanvision/blob/main/LICENSE) for details.

[issue]: https://github.com/cleanlab/cleanvision/issues/new
