import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

package_name = "cfar_plusplus"

this_dir = os.path.dirname(__file__)
requirements_path = os.path.join(this_dir, "requirements.txt")
with open(requirements_path, "r") as f:
    reqs = list(f)

setup(
    name=package_name,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1",
    description="Framework for CFAR++, a method for region-aware noise thresholding for safe radar detections.",
    # Author details
    author="Tim Bruehl",
    author_email="tim.bruehl@kit.edu",
    packages=find_packages(),
    install_requires=reqs
)
