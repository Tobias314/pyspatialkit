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
ARG CONDA_VERSION=py39_4.10.3
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



################################## Conda environment ##################################
RUN conda create -n env python=3.7

# Make RUN commands use the new environment:
RUN echo "conda activate env" >> ~/.bashrc
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
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install -n env -c open3d-admin -c conda-forge open3d -y
RUN conda install -n env geopandas -y
RUN conda install -n env tiledb-py -y
RUN conda install -n env rasterio -y
RUN conda install -n env pylint ipykernel -y
RUN conda install -n env numpy matplotlib -y
RUN conda install -n env shapely -y
RUN conda install -n env pyproj -y
RUN conda install -n env scikit-image -y
RUN conda install -n env -c conda-forge pygeos -y
RUN conda install -n env -c conda-forge fastapi uvicorn -y
RUN conda install -n env -c conda-forge genshi -y
RUN conda install -n env -c conda-forge opencv -y
RUN conda install -n env -c conda-forge eo-learn -y
RUN conda install -n env -c conda-forge pyarrow -y
RUN conda install -n env -c conda-forge pdal -y
RUN conda install -n env -c conda-forge python-pdal -y
RUN conda install -n env -c conda-forge sentinelhub -y
RUN conda install -n env -c conda-forge "shapely<1.8.0" -y
RUN conda install -n env -c conda-forge eo-learn -y
RUN conda install -n env -c conda-forge xarray dask netCDF4 bottleneck -y
RUN conda install -n env -c conda-forge autopep8 -y
RUN conda install -n env -c conda-forge trimesh -y
# RUN conda install -n env pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

################################## Pip Dependencies ##################################
RUN conda activate env; pip install pylas

################################## Npm Dependencies ##################################
RUN apt update
RUN apt install curl -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sh -c 'curl -fsSL https://deb.nodesource.com/setup_16.x | bash'
RUN apt install -y nodejs
RUN npm install -g npm
RUN npm install -g typescript
