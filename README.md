# Pyspatialkit
An open-source python library for geospatial prototyping covering management, visualizing, and parallel processing of large amounts of geospatial data with a focus on raster and point cloud data.

## Installation
There are two main ways to install PySpatialKit. First as a Docker container and second in a local Conda or Mamba environment. Installation bash scripts are provided for the Linux operating system

### Docker installation
Running `./build.sh` will build a container image named `pyspatialkit`. Running `./dev_shell_with_gui_capabilities` under Linux will open a shell inside the image and will also mount the local window server so that native visualizations (e.g. utilizing Open3D) are possible from inside the container. Alternatively, just starting a normal bash inside the container also works but lacks some visualization features.

### Conda/Mamba installation
Because of the large number of dependencies an installation using Conda might take quite long. Therefore, we use the faster Mamba as a Conda replacement. When you already use a Mamba environment just run `./install_mamba.sh` inside your environment. Otherwise run `./install_condal.sh` inside a Conda environment. Under the hood `./install_conda.sh` first installs Mamba and then runs `install_mamba.sh`.

## Usage
Until now there exists no in-depth documentation. Contributions creating more in-depth documentation are welcome. For now, this document provides an overview of the most important aspects. For more detailed information it is referred to the source code.

A good high-level understanding is provided by a simple tree detection example implemented in `examples/tree_example.py`. In addition, some high-level concepts of PySpatialKit will be described.

## Tree detection example
The example uses terrain and surface model point clouds to compute corresponding height maps as well as the height above ground by subtracting those height maps. In addition, orthophotos with infrared information are used to compute per-pixel NDVI values. Vegetation pixels high enough above the ground are classified as trees and a binary tree mask is created. Under `examples/tree_example.py` an implementation of the workflow described above can be found which covers all key parts of PySpatialKit.

There are three main sections:

1. Integrating all files into a `GeoStorage` containing three layers (two `GeoPointCloudLayers` for elevation and surface point clouds and one `GeoRasterLayer` for rgbi raster data) 
2. Defining a layer processor method that takes a spaced descriptor in form of a `GeoBox2d` as well as the input layers and processes one tile (described by the `GeoBox2d`)
3. Applying the layer processor for parallelized processing of a large area of interest (AOI)

The following sections describe the PySpatialKit features used to implement the three sections:

### Creation of GeoStorage
The `/examples` directory contains a shapefile covering an AOI in the state of Saxony-Anhalt, Germany. Point clouds and rgbi images are downloaded automatically from the open geodata portal of Saxony-Anhalt using a small script (not part of PySpatialKit). By modifying `examples/example_aoi.shp` any AOI within the border of Saxony-Anhalt can be used (if you change the AOI make sure to delete the `examples/tree_detection_data` directory because otherwise the cached results for the previous AOI are used).

After downloading the data three directories exist containing either .xyz point cloud files or .tif raster files. To integrate the data a `GeoStorage` is created. This is be done by creating a `GeoStorage` object with a directory path as a parameter. All data will then be stored inside this directory. Subsequently, one raster and two point cloud layers are added to the storage. This is done with the methods `add_raster_layer(...)` and `add_point_cloud_layer(...)` which take configuration parameters like the number of raster bands, or the data types used for the data. An important parameter, which you will find at many places in PySpatialkit is the coordinate reference system (CRS). PySpatialKit implements its own `GeoCrs` class which is used to provide spatial references for geodata objects. Each add_layer call returns a layer object inheriting from `GeoLayer` (`GeoRasterLayer`, `GeoPointCloudLayer`). Those layers provide the methods `write_data(in_memroy_geodata_object)` to write an in-memory object to the persistent layer and `get_data(space_descriptor)` to retrieve an in-memory representation for a given region described by a space descriptor. To populate our created `GeoLayers` we iterate over the files in the data directory, load the files into memory using geodata objects of PySpatialKit (`GeoRaster` for raster data, `GeoPointCloud` for point cloud data), and save those objects in the corresponding layer.

### Layer processor method
In PySpatialKit workflows are defined as methods operating on a single tile (and its in-memory geodata objects). This facilitates the prototyping of workflows using small example data objects. To scale those workflows a workflow method has to accept a space descriptor (`GeoBox2d`) as its first parameter. Other parameters are free but should contain `GeoLayer` objects which are used to provide input data and write results back. Finally, a layer processor is annotated with the `@layerprocessor` annotation. The exemplaric `tree_detection` method can be divided into 3 typical parts:

1. Loading the geodata objects for the current tile from the storage layers. In the example, the tile is described by `box` and two point clouds (`dom_pc, dgm_pc`) as well as one raster image (`rgb_raster`) are loaded.

2. The actual processing is now done on the in-memory geodata objects (`GeoRaster`, `GeoPointCloud` objects). In the example, the `project_to_georaster` method of `GeoPointCloud` objects is used to create height maps which are subtracted to compute the height above ground. The vegetation index is computed on the NumPy array holding the image data accessible via the `.data` property of a `GeoRaster`. 

3. Finally the resulting `GeoRaster` is written back to the result layer.


### Scaling the Workflow to a Large AOI
The only thing left to deploy the workflow not only for a single tile but for a large area is a tiler object implementing the `AbstractTiler` interface. For our example, we use a `GeoBoxTiler2d` which splits a given AOI in a grid of rectangular tiles of a given size (we use 1000m). Now we can call our `tree_detection` layer processor specifying the tiler and a fixed number of workers in the first pair of parentheses and all method parameters (except for a space descriptor because this will be provided by the tiler) in the second pair of parentheses. This subdivides the AOI into easy to handle tiles and starts 10 parallel processes each processing a subset of tiles and writing the results back to the result layer

### Visualizing a GeoStorage
PySpatialKit allows interactive visualization of large geo data sets in the web browser. To visualize an existing `GeoStorage` the only thing required is to call the `start_server` method with a `GeoStorage` object. This starts a webserver. Opening the /static path (e.g. [http://localhost:8080/static/](http://localhost:8080/static/) for the tree detection example) in a web browser provides a 3D visualization of all layers of the storage. Executing `examples/tree_example_visualization.py` starts the visualization server for `GeoStorage` of the tree detection example.

## Overview of key components
The example above already introduced most of the central components of PySpatialKit. Here we give another, structured, high-level overview. 
There are two central packages for data management, `dataobjects` for in-memory georeferenced objects and `storage` for large, persistent, georeferenced layers. Space descriptor (`GeoBox3d`, `GeoBox2d`, `GeoRect` in the `spacedescriptors` package) are used to describe 2D and 3D spatial regions and query `GeoLayers`.

### Dataobjects
All data objects have in common that they represent common spatial data types (raster image, point cloud, mesh ...) together with a geospatial referenced (`GeoCRS`). They provide methods for loading and storing data from and to files. In addition, several conversion functions exist to transform between different data types or libraries. E.g. point clouds can be projected to images (`point_cloud.project_to_georaster(...)`) or converted to Open3D point cloud  objects (`point_cloud.to_o3d`) to use functionality provided by the Open3D library. In addition, many data objects provide simple visualization methods rendering the object (some of which require GUI capabilities as described in the installation section).

### Storage
A `GeoStorage` object contains a set of `GeoLayers`. At the moment `GeoLayers` exist for raster `GeoRasterLayer` and point cloud `GeoPointCloudLayer` data. Every layer is georeferenced and contains methods for writing (`write_data`) and reading (`get_data`) data. E.g. for a `GeoPointCloud` layer the `get_data` method requires a `GeoBox3D` and returns a `GeoPointCloud` object containing all points inside the axis-aligned bounding box described by the `GeoBox3D` object. The `write_data` method used to write data to a `GeoPointCloudLayer` takes a `GeoPointCloud` object and writes it to the layer. Every `GeoLayer` is persisted in a directory on the file system.

### Tiling and processing
As shown in the tree detection example, tiler objects implementing the `AbstractTiler` interface are used to tile large AOIs into smaller, easy-to-process tiles described by space descriptors. 

It is generally advised to develop workflows on small example data first. For example, opening a jupyter notebook, loading some small point clouds and raster data in form of `GeoPointCloud` and `GeoRaster` objects, and plotting them might be the first steps for developing an analysis workflow. The `processing` package provides some utility methods for processing data objects (e.g. a simple plane detection in point clouds). In addition, all data objects integrate well with common libraries of the Python scientific stack. For example, every `GeoRaster` holds its `data` as a NumPy array allowing image processing with OpenCV and scikit-image, and every `GeoPointCloud`can be accessed as a Pandas data frame (`pc.data`) or converted to an Open3D point cloud (`pc.to_o3d()`). After a workflow has been developed on example data the code can be wrapped in a method and annotated with the `@layerprocessor` annotation to scale the processing to a large AOI as shows in the tree detection workflow.

## Development Guide

### Python Library Development
When using a docker container opening the PySpatialKit repository inside Visual Studio Code with the "Remote - Containers" extension installed will give an option to open the project inside a container. Inside the container running `pip install -e .` once helps to install PySpatialKit in edit mode so that changes to PySpatialKit source code are seen by other packages importing PySpatialKit.

In addition, some test cases for automated testing (mostly integration tests) exist under `/testing`. Contributions providing further test cases are welcome.

### Front End Development
We use webpack for compiling and packaging the frontend web application.
Running `npx webpack build` inside the `frontend/spatialkitcesium` directory packages the frontend and creates a `public` directory which can then be served by the Python server. However, for development and debugging a separation of the Python backend server and a development fronted server which automatically reloads when the frontend code changes might facilitate the development process. Therefore, start some form of backend server using PySpatialKit (e.g. using the `start_server(geo_sotrage)` method with a `GeoStorage` object as a parameter). Then change the `BACKEND_URL` constant  in `frontend/spatialkitcesium/src/app/constants.tsx` to the URL of your Python backend server and start a developer server e.g. using the following command:

`npx webpack-dev-server --port 8081`


In addition, the Visual Studio Code Chrome target might be used for debugging using the Chrome browser.
