"""Package installation setup."""
import os
import re
from pathlib import Path
from typing import Match, cast

from setuptools import find_packages, setup

_DIR = Path(__file__).parent
_PACKAGE_NAME = "wrapdisc"

setup(
    name=_PACKAGE_NAME,
    author="Ouroboros Chrysopoeia",
    author_email="impredicative@users.noreply.github.com",
    version=cast(Match, re.fullmatch(r"refs/tags/v?(?P<ver>\S+)", os.environ["GITHUB_REF"]))["ver"],  # Ex: GITHUB_REF="refs/tags/1.2.3"; version="1.2.3"
    description="Optimize both discrete and continuous variables using just a continuous optimizer such as scipy.optimize",
    keywords="optimization discrete continuous scipy.optimize wrapper",
    long_description=(_DIR / "README.md").read_text().strip(),
    long_description_content_type="text/markdown",
    url="https://github.com/impredicative/wrapdisc/",
    packages=find_packages(exclude=["scripts"]),
    python_requires=">=3.10",
    classifiers=[  # https://pypi.org/classifiers/
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
