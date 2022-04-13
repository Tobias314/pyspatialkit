mamba init
mamba config --add channels conda-forge
mamba config --set channel_priority strict
mamba install -c conda-forge tiledb=2.6.0 -y
mamba install -c conda-forge tiledb-py=0.12.0 -y
# mamba install -c conda-forge libgdal=3.4 -y
mamba install -c conda-forge geopandas -y
mamba install -c open3d-admin -c conda-forge open3d -y
mamba install -c conda-forge rasterio -y
mamba install -c conda-forge pylint ipykernel -y
mamba install -c conda-forge numpy matplotlib -y
mamba install -c conda-forge shapely -y
mamba install -c conda-forge pyproj -y
mamba install -c conda-forge scikit-image -y
mamba install -c conda-forge pygeos -y
mamba install -c conda-forge fastapi uvicorn -y
mamba install -c conda-forge genshi -y
mamba install -c conda-forge opencv -y
mamba install -c conda-forge eo-learn -y
mamba install -c conda-forge pyarrow -y
mamba install -c conda-forge pdal -y
mamba install -c conda-forge python-pdal -y
mamba install -c conda-forge sentinelhub -y
# mamba install -c conda-forge "shapely<1.8.0" -y
mamba install -c conda-forge shapely -y
mamba install -c conda-forge eo-learn -y
mamba install -c conda-forge xarray dask netCDF4 bottleneck -y
mamba install -c conda-forge autopep8 -y
mamba install -c conda-forge trimesh -y
mamba install -c conda-forge tensorboard -y
mamba install -c conda-forge mapbox_earcut -y
mamba install -c conda-forge ipympl -y
mamba install -c conda-forge loky -y
mamba install -c conda-forge cachetools -y
mamba install -c conda-forge fcl -y
#  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

################################## Pip Dependencies ##################################
pip install pylas
pip install triangle
pip install loky

################################## Npm Dependencies ##################################
mamba install -c conda-forge nodejs -y 
npm install -g npm
npm install -g typescript

