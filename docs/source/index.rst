.. image:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanvision_logo_open_source_transparent.png
  :width: 800
  :alt: CleanVision

Documentation
=======================================

CleanVision automatically detects various issues in image datasets, such as images that are: (near) duplicates, blurry,
over/under-exposed, etc. This data-centric AI package is designed as a quick first step for any computer vision project
to find problems in your dataset, which you may want to address before applying machine learning.


Installation
------------

.. tabs::

   .. tab:: pip

      .. code-block:: bash

         pip install cleanvision

      To install the package with all optional dependencies:

      .. code-block:: bash

         pip install "cleanvision[all]"

   .. tab:: source

      .. code-block:: bash

         pip install git+https://github.com/cleanlab/cleanvision.git

      To install the package with all optional dependencies:

      .. code-block:: bash

         pip install "git+https://github.com/cleanlab/cleanvision.git#egg=cleanvision[all]"




How to Use CleanVision
----------------------

Basic Usage
^^^^^^^^^^^
Here's how to quickly audit your image data:


.. code-block:: python3

    from cleanvision import Imagelab

    # Specify path to folder containing the image files in your dataset
    imagelab = Imagelab(data_path="FOLDER_WITH_IMAGES/")

    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()

    # Produce a neat report of the issues found in your dataset
    imagelab.report()

Targeted Issue Detection
^^^^^^^^^^^^^^^^^^^^^^^^
You can also focus on specific issues:

.. code-block:: python3

    issue_types = {"light": {}, "blurry": {}}

    imagelab.find_issues(issue_types)

    # Produce a report with only the specified issue_types
    imagelab.report(issue_types.keys())

Integration with Hugging Face Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Easily use CleanVision with a Hugging Face dataset:

.. code-block:: python3

    from datasets import load_dataset, concatenate_datasets

    # Download and concatenate different splits
    dataset_dict = load_dataset("cifar10")
    dataset = concatenate_datasets([d for d in dataset_dict.values()])

    # Specify the key for Image feature in dataset.features in `image_key` argument
    imagelab = Imagelab(hf_dataset=dataset, image_key="img")

    imagelab.find_issues()

    imagelab.report()

Integration with Torchvision Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
CleanVision works smoothly with Torchvision datasets too:


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


Additional Resources
--------------------
- Get started with our `Example Notebook <https://cleanvision.readthedocs.io/en/latest/tutorials/tutorial.html>`_
- Explore more `Example Notebooks <https://github.com/cleanlab/cleanvision-examples>`_
- Learn how to contribute in the `Contribution Guide <https://github.com/cleanlab/cleanvision/blob/main/CONTRIBUTING.md>`_


.. toctree::
   :hidden:

   Quickstart <self>


.. _tutorials:
.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Tutorials
   :name: _tutorials

   How to Use CleanVision <tutorials/tutorial.ipynb>
   tutorials/torchvision_dataset.ipynb
   tutorials/huggingface_dataset.ipynb
   Frequently Asked Questions <faq>

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
   :caption: Links
   :hidden:

   Website <https://cleanlab.ai/>
   GitHub <https://github.com/cleanlab/cleanvision.git>
   PyPI <https://pypi.org/project/cleanvision/>
   Cleanlab Studio <https://cleanlab.ai/studio/?utm_source=cleanvision&utm_medium=docs&utm_campaign=clostostudio>

