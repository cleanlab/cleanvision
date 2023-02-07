.. cleanvision documentation master file, created by
   sphinx-quickstart on Thu Feb  2 13:55:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cleanvision's documentation
=======================================
CleanVision automatically detects various issues in image datasets, such as images that are: (near) duplicates, blurry,
over/under-exposed, etc. This data-centric AI package is designed as a quick first step for any computer vision project
to find problems in your dataset, which you may want to address before applying machine learning.


Installation
------------

To install cleanvision using pip:

.. code-block:: console

   (.venv) $ pip install cleanvision

Quickstart
----------------

Using CleanVision to audit your image data is as simple as running the code below:

.. code-block:: python3

    from cleanvision.imagelab import Imagelab

    # Specify path to folder containing the image files in your dataset
    imagelab = Imagelab(path)

    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()

    # Produce a neat report of the issues found in your dataset
    imagelab.report()

.. toctree::
   :hidden:

   Quickstart <self>

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API Reference:

   cleanvision/imagelab
   cleanvision/issue_managers/image_property
   cleanvision/issue_managers/image_property_issue_manager
   cleanvision/issue_managers/duplicate_issue_manager
   cleanvision/utils/base_issue_manager
   cleanvision/utils/viz_manager
   cleanvision/utils/utils
