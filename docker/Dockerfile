# https://github.com/scientificcomputing/packages/blob/main/fenics/Dockerfile

# Dependency versions
ARG PYBIND11_VERSION=2.10.1
ARG PETSC_VERSION=3.18.6
ARG SLEPC_VERSION=3.18.3
ARG UFL_BRANCH="main"
ARG DOLFIN_BRANCH="master"
ARG FFC_BRANCH="master"
ARG DIJITSO_BRANCH="master"
ARG FIAT_BRANCH="master"

# Flags to pass to ninja build system.
ARG MAKEFLAGS
# Compiler optimisation flags for SLEPc and PETSc, all languages.
# CMake build type for DOLFIN C++ build. See CMake documentation.
ARG DOLFIN_CMAKE_BUILD_TYPE="RelWithDebInfo"
# Extra CMake C++ compiler flags for DOLFIN C++ build.
ARG DOLFIN_CMAKE_CXX_FLAGS="-O2"

########################################


FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as cuda-fenics-base


# Usage of pip setuptools
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# PETSC vars
ENV PETSC_DIR=/usr/local/petsc
ENV PETSC_ARCH=linux-gnu-real-32
# SLEPC vars
ENV SLEPC_DIR=/usr/local/slepc

# Arguments that can be overridden by a user building the docker image
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2"
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"
# MPI variant. "mpich" or "openmpi".
ARG MPI="mpich"
ARG PETSC_VERSION
ARG SLEPC_VERSION
ARG PYBIND11_VERSION

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
WORKDIR /tmp


# Install dependencies available via apt-get.
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    clang \
    cmake \
    g++ \
    git \
    gfortran \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-math-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-timer-dev \
    libeigen3-dev \
    libhdf5-${MPI}-dev \
    liblapack-dev \
    lib${MPI}-dev \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-venv \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install Python packages (via pip)
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install Cython==0.29.32
RUN python3 -m pip install pytest pkgconfig
RUN python3 -m pip install --no-cache-dir mpi4py

# Install pybind11
RUN python3 -m pip install --no-cache-dir pybind11==${PYBIND11_VERSION} && \
    wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \    
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make && \
    make install



# Install PETSc with real and complex types

RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    # Real, 32-bit int
    cd ${PETSC_DIR} && \
    python3 ./configure \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-blacs \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make ${MAKEFLAGS} all && \
    # Install petsc4py
    cd src/binding/petsc4py && \
    python3 -m pip install --no-cache-dir . && \
    # Cleanup
    apt-get -y purge bison flex && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf \
    ${PETSC_DIR}/**/tests/ \
    ${PETSC_DIR}/**/obj/ \
    ${PETSC_DIR}/**/externalpackages/  \
    ${PETSC_DIR}/CTAGS \
    ${PETSC_DIR}/RDict.log \
    ${PETSC_DIR}/TAGS \
    ${PETSC_DIR}/docs/ \
    ${PETSC_DIR}/share/ \
    ${PETSC_DIR}/src/ \
    ${PETSC_DIR}/systems/ && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install SLEPc
RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    python3 ./configure && \
    make SLEPC_DIR=${SLEPC_DIR} && \
    # Install slepc4py
    cd src/binding/slepc4py && \
    python3 -m pip install --no-cache-dir . && \
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*


FROM cuda-fenics-base as cuda-fenics-end-user
LABEL description="End user image"

ARG FIAT_BRANCH
ARG FFC_BRANCH
ARG UFL_BRANCH
ARG DOLFIN_BRANCH
ARG DIJITSO_BRANCH
ARG DOLFIN_CMAKE_BUILD_TYPE
ARG DOLFIN_CMAKE_CXX_FLAGS
ARG MAKEFLAGS

WORKDIR /src
ENV PETSC_DIR=/usr/local/petsc
ENV PETSC_ARCH=linux-gnu-real-32
ENV ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

RUN git clone --branch ${FIAT_BRANCH} --single-branch  https://github.com/FEniCS/fiat.git && \
    git clone --branch ${FFC_BRANCH} --single-branch https://bitbucket.com/fenics-project/ffc.git && \
    git clone --branch ${UFL_BRANCH} --single-branch https://github.com/FEniCS/ufl-legacy.git && \
    git clone --branch ${DOLFIN_BRANCH} --single-branch https://bitbucket.com/fenics-project/dolfin.git && \
    git clone --branch ${DIJITSO_BRANCH} --single-branch https://bitbucket.com/fenics-project/dijitso.git

RUN cd fiat && python3 -m pip install --no-cache-dir . && \
    cd ../ufl-legacy && python3 -m pip install --no-cache-dir . && \
    cd ../dijitso && python3 -m pip install --no-cache-dir . && \
    cd ../ffc && python3 -m pip install --no-cache-dir . && \
    cd ../ && python3 -m pip install --no-cache-dir ipython

RUN cd dolfin && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFIN_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFIN_CMAKE_CXX_FLAGS} -B build . && \
    ninja ${MAKEFLAGS} install -C build

RUN python3 -m pip install --no-dependencies ./dolfin/python

WORKDIR /root
RUN rm -rf /src


# https://github.com/scientificcomputing/packages/blob/main/fenics-gmsh/Dockerfile
# FROM ghcr.io/scientificcomputing/fenics:2023-08-14
FROM cuda-fenics-end-user AS cuda-fenics-end-user-gmsh

# Dependency versions
ARG GMSH_VERSION=4_11_1

WORKDIR /tmp

# Install dependencies available via apt-get.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev  \
    libgl1-mesa-dev \
    libocct-foundation-dev \
    libocct-data-exchange-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install OpenCASACADE
# OpenCASACADE is currently installed via apt, but the following lines
# would be way to install this from source in case this is needed later
# ARG OCC_VERSION=7_7_0
# RUN curl -L -o occt.tgz "http://git.dev.opencascade.org/gitweb/?p=occt.git;a=snapshot;h=refs/tags/V${OCC_VERSION};sf=tgz" && \
#     tar zxf occt.tgz && \
#     cmake -G Ninja -DBUILD_MODULE_Draw:BOOL=FALSE -DCMAKE_BUILD_TYPE=Release -B build-occt -S occt-V${OCC_VERSION} && \
#     cmake --build build-occt && \
#     cmake --install build-occt && \
#     rm -rf /tmp/*

# Install gmsh
RUN git clone -b gmsh_${GMSH_VERSION} --single-branch --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1 -B build-dir -S gmsh && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# GMSH installs python package in /usr/local/lib, see: https://gitlab.onelab.info/gmsh/gmsh/-/issues/1414
RUN export SP_DIR=$(python3 -c 'import sys, site; sys.stdout.write(site.getsitepackages()[0])') \
    && mv /usr/local/lib/gmsh.py ${SP_DIR}/ \
    && mv /usr/local/lib/gmsh*.dist-info ${SP_DIR}/

# build h5py from source
# disable build isolation due to https://github.com/h5py/h5py/issues/2222
RUN export HDF5_MPI=ON \
    && export HDF5_PKGCONFIG_NAME="hdf5" \
    && python3 -m pip install cython pkgconfig mpi4py \
    && python3 -m pip install --no-cache-dir --no-binary=h5py --no-build-isolation h5py -vv
RUN python3 -m pip install --no-cache-dir meshio

WORKDIR /


FROM cuda-fenics-end-user-gmsh AS cuda-fenics-adjoint

# Install dolfin-adjoint
RUN python3 -m pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git



## Install Ipopt
WORKDIR /usr/local
RUN git clone https://github.com/coin-or/Ipopt.git
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
RUN cd ThirdParty-Mumps && ./get.Mumps && ./configure --prefix=/usr/local && make && make install
RUN cd Ipopt && mkdir build && cd build && ../configure --without-hsl --without-asl --with-lapack --with-mumps --prefix=/usr/local && make && make test && make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH


# Install cyipopt
RUN /bin/bash -l -c "pip3 install Cython"
RUN /bin/bash -l -c "pip3 install six"

RUN /bin/bash -l -c "pip3 install git+https://github.com/mechmotum/cyipopt.git"



FROM cuda-fenics-adjoint AS cuda-fenics-pytorch

RUN python3 -m pip --no-cache-dir install torch torchvision torchaudio


FROM cuda-fenics-pytorch AS cuda-fenics-pytorch-pyvista

# Install pyvista-requirement vtk
RUN python3 -m pip install vtk

# Install pyvista
RUN python3 -m pip install pyvista

# Install everything else
RUN python3 -m pip install numpy matplotlib ipykernel tqdm pytest pyyaml pandas dill pygmsh

# Install gmshnics
RUN python3 -m pip install git+https://github.com/MiroK/gmshnics.git
