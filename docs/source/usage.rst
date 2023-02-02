Usage
=====

.. _installation:

Installation
------------

To install cleanvision using pip:

.. code-block:: console

   (.venv) $ pip install cleanvision


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

