# Development

This document explains how to set up a development environment for [contributing](CONTRIBUTING.md) to CleanVision.

## Setting up a virtual environment

While this is not required, we recommend that you do development and testing in
a virtual environment. There are a number of tools to do this, including
[virtualenv](https://virtualenv.pypa.io/), [pipenv](https://pipenv.pypa.io/),
and [venv](https://docs.python.org/3/library/venv.html). You can
[compare](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
the tools and choose what is right for you. Here, we'll explain how to get set
up with venv, which is built in to Python 3.

```shell
python3 -m venv ./ENV  # create a new virtual environment in the directory ENV
source ./ENV/bin/activate  # switch to using the virtual environment
```

You only need to create the virtual environment once, but you will need to
activate it every time you start a new shell. Once the virtual environment is
activated, the `pip install` commands below will install dependencies into the
virtual environment rather than your system Python installation.

## Installing dependencies and cleanvision package

Run the following commands in the repository's root directory.
1. Upgrade pip
```shell
python -m pip install --upgrade pip
```
2. Install CleanVision as an editable package
```shell
pip install -e .
```
3. Install development requirements
```shell
pip install -r requirements-dev.txt
```
4. Inorder to set the path to pytest, it is required to deactivate and activate the environment when pytest is installed.
```shell
deactivate
source ./ENV/bin/activate
```

## Testing

**Run all the tests:**

```shell
pytest
```

**Run a specific file or test:**

```shell
pytest -k <filename or filter expression>
```

**Run with verbose output:**

```shell
pytest --verbose
```

**Run with code coverage:**

```shell
pytest --cov=cleanvision --cov-config .coveragerc --cov-report=html
```

The coverage report will be available in `coverage_html_report/index.html`,
which you can open with your web browser.

## Type checking

CleanVision uses [mypy](https://mypy.readthedocs.io/en/stable/) typing. Type checking happens automatically during CI but can be run locally.

**Check typing in all files:**

```shell
mypy --strict --install-types --non-interactive --python-version 3.11  src
```

## How to style new code contributions

CleanVision follows the [Black](https://black.readthedocs.io/) code style. This is
enforced by CI, so please format your code by invoking `black` before submitting a pull request.
Use the following command to format your code.

```shell
python -m black src
```

Generally aim to follow the [PEP-8 coding style](https://peps.python.org/pep-0008/).
Please do not use wildcard `import *` in any files, instead you should always import the specific functions that you need from a module.
Use the following command to check for style errors.

```shell
flake8 --ignore=E203,E501,E722,E401,W503 src tests --count --show-source --statistics
```

### Pre-commit hook

You can also use [pre-commit framework](https://pre-commit.com/) to easily
set up code style checks that run automatically whenever you make a commit.
You can install the git hook scripts with:

```shell
pre-commit install
```

### EditorConfig

This repo uses [EditorConfig](https://editorconfig.org/) to keep code style
consistent across editors and IDEs. You can install a plugin for your editor,
and then your editor will automatically ensure that indentation and line
endings match the project style.

## Merging PRs to the codebase

The docs for this repo are hosted by readthedocs.io and can be found [here](https://cleanvision.readthedocs.io/en/latest/).


## Publishing Release
- Ensure all **Checks** are successful for your release related PR after getting approvals from other code maintainers.
- Merge all release related PRs.
- Go to the Releases in the Github repo and draft a new release.
- Create a new tag in the format `v{version}` corresponding to the version being released and click *Generate  release notes*.
- Before publishing the release, ensure all checks are satisfied as the release on TestPypi and Pypi is automated and an error will result in publishing a new version.
- Publish release
