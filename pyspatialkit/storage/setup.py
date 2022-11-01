
from setuptools import setup, find_namespace_packages

setup(
    name='pyspatialkit_storage',
    version='0.0.1',
    author='Tobias Pietz',
    author_email='pito.mailing@gmail.com',
    packages = find_namespace_packages(where='src'),
    package_dir={"": "src"},
    license='TODO',#TODO
    description='TDODO',
    long_description="TODO",#TODO'
    install_requires=['pyspatialkit_core', 'SQLAlchemy' 'fs'],#TODO
)