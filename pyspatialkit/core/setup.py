
from setuptools import setup, find_namespace_packages

setup(
    name='pyspatialkit_core',
    version='0.0.1',
    author='Tobias Pietz',
    author_email='pito.mailing@gmail.com',
    packages = find_namespace_packages(where='src'),
    package_dir={"": "src"},
    license='TODO',#TODO
    description='TDODO',
    long_description="TODO",#TODO'
    install_requires=['pyproj', 'shapely', 'numpy', 'geopandas', 'trimesh', 'python-fcl', 'open3d',
                      'rasterio', 'scipy', 'pygeos', 'scikit-image'],#TODO
)