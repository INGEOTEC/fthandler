# Copyright 2019 Eric S. Tellez <eric.tellez@infotec.mx>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
import fthandler


setup(
    name="fthandler",
    description="""A small hyper-parameter optimization module for fastText""",
    long_description="""Given a classification task, this module helps on the
    search of the best configuration for fastText. It automatically explores
    the configuration space using a k-fold cross-validation scheme.

    """,
    version=fthandler.__version__,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    packages=['fthandler'],
    include_package_data=True,
    zip_safe=False,
    package_data={
        'fthandler/': ['emojis.txt'],
    },
    scripts=[
    ]
)
