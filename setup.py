from setuptools import setup
import os
import inspect

project_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# setup for bagofvisualwords

setup(
    name='bagofvisualwords',
    version='0.0.dev0',
    author='Matthew Antalek',
    author_email='matthew.antalek@gmail.com',
    license='MIT',
    package_dir={'':'src'},
    packages=['bagofvisualwords']
)

# setup for patchifier

setup(
    name='patchifer',
    version='0.0.dev0',
    author='Matthew Antalek',
    author_email='matthew.antalek@gmail.com',
    license='MIT',
    package_dir={'':'src'},
    packages=['patchifier']
)
