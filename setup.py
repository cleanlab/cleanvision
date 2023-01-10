# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get("install", True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file("LICENSE", self.egg_info)

        egg_info.run(self)


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cleanvision",
    version="0.0.0",
    license="AGPLv3+",
    python_requires=">=3.7",
    # What does your project relate to?
    keywords="computer_vision cv image_data issue_detection data_quality image_quality machine_learning data_cleaning image_deduplication",
    packages=find_packages("src"),
    package_data={
        "": ["LICENSE"],
    },
    license_files=("LICENSE",),
    cmdclass={"egg_info": egg_info_ex},
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "Pillow>=9.3",
        "numpy>=1.20.0",
        "pandas>=1.1.5",
        "imagehash>=4.2.0",
        "tqdm>=4.53.0",
        "matplotlib>=3.4",
    ],
)
