import numpy as np
from setuptools import Extension, setup

include_path = [np.get_include()]

extensions = [
    Extension(
        "yt_experiments.octree.converter",
        ["yt_experiments/octree/converter.pyx"],
        include_dirs=include_path,
    ),
]

setup(ext_modules=extensions)
