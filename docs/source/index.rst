.. image:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanvision_logo_open_source_transparent.png
  :width: 800
  :alt: CleanVision

Documentation
=======================================
CleanVision automatically detects various issues in image datasets, such as images that are: (near) duplicates, blurry,
over/under-exposed, etc. This data-centric AI package is designed as a quick first step for any computer vision project
to find problems in your dataset, which you may want to address before applying machine learning.


Installation
============

To install the latest stable version (recommended):

.. code-block:: console

   $ pip install cleanvision


To install the bleeding-edge developer version:

.. code-block:: console

   $ pip install git+https://github.com/cleanlab/cleanvision.git

To install with HuggingFace optional dependencies

.. code-block:: console

   $ pip install "cleanvision[huggingface]"

To install with Torchvision optional dependencies

.. code-block:: console

   $ pip install "cleanvision[pytorch]"





Quickstart
===========

1. Using CleanVision to audit your image data is as simple as running the code below:


.. code-block:: python3

    from cleanvision.imagelab import Imagelab

    # Specify path to folder containing the image files in your dataset
    imagelab = Imagelab(data_path="FOLDER_WITH_IMAGES/")

    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()

    # Produce a neat report of the issues found in your dataset
    imagelab.report()

2. CleanVision diagnoses many types of issues, but you can also check for only specific issues:


.. code-block:: python3

    issue_types = {"light": {}, "blurry": {}}

    imagelab.find_issues(issue_types)

    # Produce a report with only the specified issue_types
    imagelab.report(issue_types.keys())

3. Run CleanVision on a Hugging Face dataset


.. code-block:: python3

    from datasets import load_dataset, concatenate_datasets

    # Download and concatenate different splits
    dataset_dict = load_dataset("cifar10")
    dataset = concatenate_datasets([d for d in dataset_dict.values()])

    # Specify the key for Image feature in dataset.features in `image_key` argument
    imagelab = Imagelab(hf_dataset=dataset, image_key="img")

    imagelab.find_issues()

    imagelab.report()

4. Run CleanVision on a Torchvision dataset


.. code-block:: python3

    from torchvision.datasets import CIFAR10
    from torch.utils.data import ConcatDataset

    # Download and concatenate train set and test set
    train_set = CIFAR10(root="./", download=True)
    test_set = CIFAR10(root="./", train=False, download=True)
    dataset = ConcatDataset([train_set, test_set])


    imagelab = Imagelab(torchvision_dataset=dataset)

    imagelab.find_issues()

    imagelab.report()


More on how to get started with CleanVision:

- `Tutorial Notebook <https://github.com/cleanlab/cleanvision-examples/blob/main/tutorial.ipynb>`_
- `Run CleanVision on a HuggingFace dataset <https://github.com/cleanlab/cleanvision-examples/blob/main/huggingface_dataset.ipynb>`_
- `Run CleanVision on a Torchvision dataset <https://github.com/cleanlab/cleanvision-examples/blob/main/torchvision_dataset.ipynb>`_
- `Example Python script <https://github.com/cleanlab/cleanvision/blob/main/examples/run.py>`_
- `Additional example notebooks <https://github.com/cleanlab/cleanvision-examples>`_


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   Quickstart <self>
.. _api-reference:

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API Reference
   :name: _api_reference

   cleanvision/imagelab
   cleanvision/issue_managers/index
   cleanvision/dataset/index
   cleanvision/utils/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Tutorials

   tutorials/tutorial.ipynb
   tutorials/torchvision_dataset.ipynb
   tutorials/huggingface_dataset.ipynb
   

.. toctree::
   :caption: Guides
   :hidden:

   Tutorial Notebook <https://colab.research.google.com/github/cleanlab/cleanvision/blob/main/examples/tutorial.ipynb>
   HuggingFace Notebook <https://colab.research.google.com/github/cleanlab/cleanvision/blob/main/examples/huggingface_dataset.ipynb>
   Torchvision Notebook <https://colab.research.google.com/github/cleanlab/cleanvision/blob/main/examples/torchvision_dataset.ipynb>
   Example Python Script <https://github.com/cleanlab/cleanvision/blob/main/examples/run.py>
   More Example Notebooks <https://github.com/cleanlab/cleanvision-examples>
   How To Contribute <https://github.com/cleanlab/cleanvision/blob/main/CONTRIBUTING.md>

.. toctree::
   :caption: Links
   :hidden:

   Website <https://cleanlab.ai/>
   GitHub <https://github.com/cleanlab/cleanvision.git>
   PyPI <https://pypi.org/project/cleanvision/>
