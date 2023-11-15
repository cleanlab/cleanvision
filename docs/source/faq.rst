Frequently Asked Questions
==========================

Answers to frequently asked questions about the `cleanvision <https://github.com/cleanlab/cleanvision/>`_ open-source package.

1. **What kind of machine learning tasks can I use CleanVision for?**

CleanVision is independent of any machine learning tasks as it directly works on images and does not require and labels or metadata to detect issues in the dataset. The issues detected by CleanVision are helpful for all kinds of machine learning tasks.

2. **Can I check for specific issues in my dataset?**


Yes, you can specify issues like ``light`` or ``blurry`` in the issue_types argument when calling ``Imagelab.find_issues``

.. code-block:: python3

    imagelab.find_issues(issue_types={"light": {}, "blurry": {}})


3. **What dataset formats does CleanVision support?**


Apart from plain image files, CleanVision also works with HuggingFace and Torchvision datasets. You can use the dataset objects as is with the ``image_key`` argument.

.. code-block:: python3

    imagelab = Imagelab(hf_dataset=dataset, image_key="image")

For more detailed usage instructions and examples, check the :ref:`tutorials`.

Commonly encountered errors
---------------------------

- **RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.**

.. code-block:: console

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            freeze_support()
            ...

    The "freeze_support()" line can be omitted if the program
    is not going to be frozen to produce an executable.

    To fix this issue, refer to the "Safe importing of main module"
    section in https://docs.python.org/3/library/multiprocessing.html


The above issue is caused by multiprocessing module working differently for macOS and Windows platforms. A detailed discussion of the issue can be found `here <https://github.com/cleanlab/cleanlab/issues/159>`_.
A fix around this issue is to run CleanVision in the main namespace like this

.. code-block:: python3

    if __name__ == "__main__":

        imagelab = Imagelab(data_path)
        imagelab.find_issues()
        imagelab.report()

OR use `n_jobs=1` to disable parallel processing:

.. code-block:: python3

    imagelab.find_issues(n_jobs=1)