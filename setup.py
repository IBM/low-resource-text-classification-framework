# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import setuptools
import os

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# collecting all dependencies from requirements.txt
parent_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(parent_dir, 'lrtc_lib', 'requirements.txt')
with open(req_file) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="low-resource-text-classification-framework",
    version="0.1",
    author="Project Debater",
    author_email="alonhal@il.ibm.com",
    description="Framework for text classification in low-resource scenarios",
    long_description="Research framework for low resource text classification that allows the user to experiment with "
                     "classification models and active learning strategies on a large number of sentence classification"
                     " datasets, and to simulate real-world scenarios. The framework is easily expandable to new "
                     "classification models, active learning strategies and datasets.",
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/Debater/sleuth",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
)