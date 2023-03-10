![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanvision_logo_open_source_transparent.png)

<img width="1200" alt="Screen Shot 2023-03-10 at 10 23 33 AM" src="https://user-images.githubusercontent.com/10901697/224394144-bb0e1c85-6851-4828-bcd2-4ed234270a78.png">

[![Read the Docs](https://readthedocs.org/projects/cleanvision/badge/?version=latest)](https://cleanvision.readthedocs.io/en/latest/)
 [![codecov](https://codecov.io/github/cleanlab/cleanvision/branch/main/graph/badge.svg?token=y1N6MluN9H)](https://codecov.io/gh/cleanlab/cleanvision)

CleanVision automatically detects various issues in image datasets, such as images that are: (near) duplicates,
blurry, over/under-exposed, etc. This data-centric AI package is designed as a quick first step for any computer vision
project to find problems in your dataset, which you may want to address before applying machine learning.


## Quickstart

Using CleanVision to audit your image data is as simple as running the code below:

```python

from cleanvision.imagelab import Imagelab

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(path)

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()

# Produce a neat report of the issues found in your dataset
imagelab.report()
```

CleanVision diagnoses many types of issues, but you can also check for only specific issues:

```python
issue_types = {"light": {}, "blurry": {}}

imagelab.find_issues(issue_types)

# Produce a report with only the specified issue_types
imagelab.report(issue_types.keys())
```

More on how to get started with CleanVision:

- [Example script](https://github.com/cleanlab/cleanvision/blob/main/examples/run.py)
- [Jupyter notebook tutorial](https://github.com/cleanlab/cleanvision/blob/main/examples/demo.ipynb)
- [Documentation](https://cleanvision.readthedocs.io/)

Example image dataset you can run this library on can be downloaded using:

```python
wget -nc
'https://cleanlab-public.s3.amazonaws.com/CleanVision/image_files.zip'
```


## Issues in image data

At the moment this package is a work in progress (expect sharp edges!) and can detect the following types of issues:

|     | Issue Type      | Description                                                                                  | Issue Key        |
|-----|------------------|----------------------------------------------------------------------------------------------|------------------|
| 1   | Light            | Images that are too bright/washed out in the dataset                                         | light            |
| 2   | Dark             | Images that are irregularly dark                                                             | dark             |
| 3   | Odd Aspect Ratio | Images with an unusual aspect ratio (i.e. overly skinny/wide)                                                       | odd_aspect_ratio |
| 4   | Exact Duplicates | Images that are exact duplicates of each other                          | exact_duplicates |
| 5   | Near Duplicates  | Images that are almost visually identical to each other                                 | near_duplicates  |
| 6   | Blurry           | Images that are blurry or out of focus                                                  | blurry           |
| 7   | Grayscale        | Images that are grayscale (lacking color)                                                            | grayscale        |
| 8   | Low Information  | Images that lack much information (e.g. a completely black image with a few white dots) | low_information  |

The **Issue Key** column above specifies the name for each type of issue in the CleanVision code. Use these string names to specify: which types of issues should be considered or what non-default parameter values to use when checking for certain issues.

CleanVision currently detects issues in the raw images themselves, and can be useful for any machine learning task (classification, segmentation, object detection, pose estimation, keypoint detection, etc). To detect issues in the labels of your image data, you can instead use [cleanlab](https://github.com/cleanlab/cleanlab/).

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
