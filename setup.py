from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='simple2d',
  version='0.0.1',
  description='simple 2d (runge-kutta) physics engine',
  author='2cscsc0',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=['simple2d'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License"
  ],
  install_requires=["tqdm", "numpy", "pillow", "ffmpeg-python", "scipy", "matplotlib"],
  python_requires='>=3.10',
  extras_require={
    "linting": [
      "pylint",
      "mypy",
      "typing-extensions",
      "types-tqdm",
      "types-pillow",
    ],
  },
  include_package_data=True,
)
