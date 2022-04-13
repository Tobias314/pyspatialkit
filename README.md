# Pyspatialkit
An open source python library for geospatial prototyping covering management, visualizing and processing of large amounts of geospatial data with a focus on raster and point cloud data.

## Installation
There are two main ways to install PySpatialKit. First as a Docker container and second in a local Conda or Mamba environment. Installation bash scripts are provided for the linux operating system

### Docker installation
Running `./build.sh` will build a container image named `pyspatialkit`. Running `./dev_shell_with_gui_capabilities` under Linux will open a shell inside the image and will also mount the local window server so that native visualizations (e.g. utilizing Open3D) are possible from inside the container. Alternatively, just starting a normal bash inside the container also works but lacks some visualization features.

### Conda/Mamba installation
Because of the large number of dependencies an installation using Conda might take quite long. Therefore, we use the faster Mamba as a conda replacement. When you already use a Mamba environment just run `./install_mamba.sh` inside your environment. Otherwise run `./install_condal.sh` inside a Conda environment. Under the hood `./install_conda.sh` first installs Mamba and then runs `install_mamba.sh`.

## Usage


## Development Guide
### Front End Development
Start the dev server in frontend/spatialkitcesion using:

`npx webpack-dev-server --port 8081`

Start some form of backend server using pyspatialkit (e.g. using the `start_server(geo_sotrage)` method with a GeoStorage object as parameter)

Then use the VScode target Chrome to debug using chrome
