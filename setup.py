from setuptools import setup, find_packages

setup(
    name='tcn-keras',
    version='0.0.1',
    description='Keras-based Temporal Convolutional Network implementation',
    author='Jakob Struye',
    license='MIT',
    packages=find_packages(),
    install_requires=['keras', 'numpy']
)
