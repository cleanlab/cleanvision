# Development

This document explains how to set up a development environment for contributing to clean-vision.

## Setting up a virtual environment

While this is not required, we recommend that you do development and testing in
a virtual environment. There are a number of tools to do this, including
[virtualenv](https://virtualenv.pypa.io/), [pipenv](https://pipenv.pypa.io/),
and [venv](https://docs.python.org/3/library/venv.html). You can
[compare](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
the tools and choose what is right for you. Here, we'll explain how to get set
up with venv, which is built in to Python 3.

```console
$ python3 -m venv ./ENV  # create a new virtual environment in the directory ENV
$ source ./ENV/bin/activate  # switch to using the virtual environment
```

You only need to create the virtual environment once, but you will need to
activate it every time you start a new shell. Once the virtual environment is
activated, the `pip install` commands below will install dependencies into the
virtual environment rather than your system Python installation.

## Installing dependencies and clean-vision

Run the following commands in the repository's root directory.

1. Install development requirements with `pip install -r requires.txt`

2. Install clean-vision as an editable package with `pip install -e .`

## Testing

**Run all the tests:**

```console
$ pytest
```

**Run a specific file or test:**

```
$ pytest -k <filename or filter expression>
```

**Run with verbose output:**

```
$ pytest --verbose
```

## How to style new code contributions

clean-vision follows the [Black](https://black.readthedocs.io/) code style. This is
enforced by CI, so please format your code by invoking `black` before submitting a pull request.

Generally aim to follow the [PEP-8 coding style](https://peps.python.org/pep-0008/). 
Please do not use wildcard `import *` in any files, instead you should always import the specific functions that you need from a module.


## Initial Dev notes [no longer strongly applicable]
Please ask questions about this library and debugging to any of: 
Jonas, Yiming Chen (in Slack or via email at: <yimingc@mit.edu> <yiming.chen699@gmail.com>)

Library has rough edges since its development was abruptly halted.

Initial development should first focus on getting the library in a good state again and addressing basic code quality issues.


Two older versions of the library that may work better are:

1. the `august` branch of this repo (do not delete this branch for posterity).
2. the code in Google Drive (newer than august branch, this was final code Yiming ever touched, newer than august branch version):
https://drive.google.com/drive/folders/16wJPl8W643w7Tp2J05v3OMu8EpECkXpD?usp=share_link
