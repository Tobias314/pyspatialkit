FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 as dependencies


################################## Conda setup ##################################
LABEL maintainer="Anaconda, Inc"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#hadolint ignore=DL3008
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH
CMD [ "/bin/bash" ]
# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=latest
RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="1faed9abecf4a4ddd4e0d8891fc2cdaa3394c51e877af14ad6b9d4aadb4e90d8"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="4879820a10718743f945d88ef142c3a4b30dfc8e448d1ca08e019586374b773f"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
        SHA256SUM="fa92ee4773611f58ed9333f977d32bbb64769292f605d518732183be1f3321fa"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


################################## Install Mamba (faster conda replacement) ##################################
RUN conda install mamba -n base -c conda-forge
RUN mamba init
SHELL ["bash", "-lc"]


################################## Conda environment ##################################
RUN mamba create -n env python=3.7

# Make RUN commands use the new environment:
RUN echo "mamba activate env" >> ~/.bashrc
#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
SHELL ["/bin/bash", "--login", "-c"]


################################## GUI capabilities ##################################
# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

################################## Conda Dependencies ##################################
# Conda
RUN mamba config --add channels conda-forge
RUN mamba config --set channel_priority strict
RUN mamba install -n env -c conda-forge tiledb=2.6.0 -y
RUN mamba install -n env -c conda-forge tiledb-py=0.12.0 -y
#RUN mamba install -n env -c conda-forge libgdal=3.4 -y
RUN mamba install -n env -c conda-forge geopandas -y
RUN mamba install -n env -c open3d-admin -c conda-forge open3d -y
RUN mamba install -n env -c conda-forge rasterio -y
RUN mamba install -n env -c conda-forge pylint ipykernel -y
RUN mamba install -n env -c conda-forge numpy matplotlib -y
RUN mamba install -n env -c conda-forge shapely -y
RUN mamba install -n env -c conda-forge pyproj -y
RUN mamba install -n env -c conda-forge scikit-image -y
RUN mamba install -n env -c conda-forge pygeos -y
RUN mamba install -n env -c conda-forge fastapi uvicorn -y
RUN mamba install -n env -c conda-forge genshi -y
RUN mamba install -n env -c conda-forge opencv -y
RUN mamba install -n env -c conda-forge eo-learn -y
RUN mamba install -n env -c conda-forge pyarrow -y
RUN mamba install -n env -c conda-forge pdal -y
RUN mamba install -n env -c conda-forge python-pdal -y
RUN mamba install -n env -c conda-forge sentinelhub -y
#RUN mamba install -n env -c conda-forge "shapely<1.8.0" -y
RUN mamba install -n env -c conda-forge shapely -y
RUN mamba install -n env -c conda-forge eo-learn -y
RUN mamba install -n env -c conda-forge xarray dask netCDF4 bottleneck -y
RUN mamba install -n env -c conda-forge autopep8 -y
RUN mamba install -n env -c conda-forge trimesh -y
RUN mamba install -n env -c conda-forge tensorboard -y
RUN mamba install -n env -c conda-forge mapbox_earcut -y
RUN mamba install -n env -c conda-forge ipympl -y
RUN mamba install -n env -c conda-forge loky -y
RUN mamba install -n env -c conda-forge cachetools -y
# RUN conda install -n env pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

################################## Pip Dependencies ##################################
RUN conda activate env; pip install pylas
RUN conda activate env; pip install triangle
RUN conda activate env; pip install loky
RUN conda activate env; pip install python-fcl
RUN conda activate env; pip install fasteners

################################## Npm Dependencies ##################################
RUN apt update
RUN apt install curl -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sh -c 'curl -fsSL https://deb.nodesource.com/setup_16.x | bash'
RUN apt install -y nodejs
RUN npm install -g npm
RUN npm install -g typescript

COPY pyspatialkit /pyspatialkit/pyspatialkit
COPY setup.py /pyspatialkit/setup.py
RUN conda activate env; pip install -e pyspatialkit/
