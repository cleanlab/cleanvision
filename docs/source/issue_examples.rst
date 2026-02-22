.. image:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanvision_logo_open_source_transparent.png
  :width: 800
  :alt: CleanVision

Issue Examples Gallery
======================================

This page showcases a **continuously growing** list of compelling examples for each type of image issue detected by CleanVision. Images are organized by issue type; each caption includes dataset, label, and other metadata where available.

**Image storage:** Example images are stored in the `cleanlab/assets <https://github.com/cleanlab/assets/tree/master/cleanvision/example_issue_images>`_ repository (not in the CleanVision repo). To add more examples, open a PR there with new images—e.g. from running CleanVision on new datasets, `Kaggle <https://www.kaggle.com/search?q=cleanvision>`_, or posts like `CleanVision for image data cleaning <https://www.joaoataide.com/post/cleanvision-uma-ferramenta-essencial-para-limpeza-de-dados-de-imagem>`_.

.. contents::
   :local:
   :backlinks: none

Blurry
------

Images that are out of focus or have motion blur, making it difficult to extract meaningful features.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/blurry.png
   :alt: Blurry image example
   :width: 300

   Example of blurry image detected by CleanVision. Add more examples from your datasets via the assets repo.

Dark
----

Images that are underexposed or have very low brightness, making it difficult to discern content.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/dark.jpg
   :alt: Dark image example
   :width: 300

   Example of dark/underexposed image. Dataset/label can be added in caption when contributing new images.

Light
-----

Images that are overexposed or too bright, causing loss of detail in bright areas.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/light.jpg
   :alt: Light image example
   :width: 300

   Example of light/overexposed image. Add compelling examples with dataset and label in caption.

Grayscale
---------

Images that lack color information. May be intentionally grayscale or incorrectly converted.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/grayscale.jpg
   :alt: Grayscale image example
   :width: 300

   Example of grayscale image. Contribute more with metadata (dataset, label).

Low Information
---------------

Images that contain very little visual information—e.g. simple graphics, blank images, or non-photographic content.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/low_information.png
   :alt: Low information image example
   :width: 300

   Example of low-information image. Add examples from Caltech-256, CUB-200-2011, CIFAR-10, etc. with captions.

Odd Aspect Ratio
----------------

Images with unusual width-to-height ratios that may cause issues in standard model training pipelines.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/odd_aspect_ratio.jpg
   :alt: Odd aspect ratio example
   :width: 300

   Example of odd aspect ratio. Include dataset and label when adding more.

Odd Size
--------

Images with unusual dimensions that may cause processing issues.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/odd_size.png
   :alt: Odd size image example
   :width: 300

   Example of odd size image. Add more with metadata in caption.

Exact Duplicates
----------------

Images that are pixel-identical copies of each other.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/exact_duplicates.png
   :alt: Exact duplicate pair example
   :width: 400

   Example pair of exact duplicates. When adding separate image pairs, use captions like: Dataset: Caltech-256 | Label: class_name.

Near Duplicates
---------------

Images that are highly similar but not pixel-identical—e.g. different lighting, crops, or slight edits.

.. figure:: https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/near_duplicates.png
   :alt: Near duplicate pair example
   :width: 400

   Example pair of near duplicates. Add pairs with Dataset/Label (e.g. CUB-200-2011, Food-101) in captions.

Adding more examples
--------------------

To grow this gallery:

1. **Store images** in `cleanlab/assets <https://github.com/cleanlab/assets/tree/master/cleanvision/example_issue_images>`_ under ``cleanvision/example_issue_images/``. Use descriptive names (e.g. ``blurry_food101_burrito.jpg``, ``dark_2.jpg``).
2. **Update this file** (``docs/source/issue_examples.rst``) in the CleanVision repo: add a ``.. figure::`` block pointing to the raw URL ``https://raw.githubusercontent.com/cleanlab/assets/master/cleanvision/example_issue_images/<filename>`` and a caption with dataset, label, and any other metadata.
3. **Sources for images:** Run CleanVision on public datasets (e.g. Food-101, CIFAR-10, Caltech-256, CUB-200-2011), search `Kaggle <https://www.kaggle.com/search?q=cleanvision>`_, or use examples from blog posts and tutorials. We only need the **most compelling** examples per issue type, not every issue from every dataset.

.. raw:: html

   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/cleanlab/cleanvision" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star cleanlab/cleanvision on GitHub">Star CleanVision</a>
