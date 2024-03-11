# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="liquid_networks",
    version="1.0.0",
    author="Samuel Berrien",
    packages=find_packages(include=["liquid_networks", "liquid_networks.*"]),
)
