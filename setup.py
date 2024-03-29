import os
from codecs import open
from setuptools import setup, find_packages
packages = ['feature_engineering']
requires = ['numpy', 'pandas']
about = {}
with open(os.path.join('__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    #long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=find_packages(exclude=('tests', 'tests.*')),
    #package_dir={'feature_engineering': 'feature_engineering'},
    include_package_data=True,
    python_requires=">=3.5, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    install_requires=requires,
    license=about['__license__'],
    zip_safe=False
)
