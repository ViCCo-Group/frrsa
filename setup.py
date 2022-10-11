try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
                'joblib==1.2.*',
                'numpy==1.23.*',
                'numba==0.56.*',
                'pandas==1.5.*',
                'psutil==5.9.*',
                'scikit-learn==1.1.*',
                'scipy==1.9.*'
                ]

setuptools.setup(
                name="frrsa",
                version="0.0.1",
                author="Philipp Kaniuth",
                author_email="kaniuth@cbs.mpg.de",
                description="Python package to conduct feature-reweighted representational similarity analysis.",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/ViCCo-Group/frrsa",
                packages=setuptools.find_packages(),
                license="GNU AFFERO GENERAL PUBLIC LICENSE Version 3",
                install_requires=requirements,
                keywords="feature extraction",
                classifiers=[
                    "Programming Language :: Python :: 3.8",
                    "Natural Language :: English",
                    "License :: OSI Approved :: AGPL-3.0",
                    "Operating System :: OS Independent",
                ],
                python_requires='>=3.8',
)
