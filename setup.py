from __future__ import print_function, unicode_literals
from distutils.core import setup

setup(
    name='MeasureDetector',
    version='0.1',
    packages=['MeasureDetector', 'MeasureDetector.configurations', 'MeasureDetector.demo'],
    url='https://github.com/omr-research/measuredetector',
    license='MIT',
    author='Alexander Pacha, Simon Waloschek',
    author_email='alexander.pacha@tuwien.at',
    description='A deep-learning based detector for measures in musical scores built on top of the Tensorflow Object Detection API'
)
