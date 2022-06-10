FROM finsberg/py310-base

RUN python -m pip install "numpy<1.22"

RUN apt-get update \
	&& apt-get install -y wget \
	&& rm -rf /var/lib/apt/lists/*

##################
# Download ipoptr source
# See these nasty files:
# http://www.coin-or.org/Ipopt/documentation/node10.html
##################
RUN wget http://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.6.tgz && \
    gunzip Ipopt-3.12.6.tgz && \
    tar -xvf Ipopt-3.12.6.tar && \
    rm -rf Ipopt-3.12.6.tar && \
    mv Ipopt-3.12.6 CoinIpopt && \
    cd CoinIpopt && \
# Downloading BLAS, LPACK, and ASL
    cd ThirdParty/ASL && \
        ./get.ASL && \
    cd ../Blas && \
        ./get.Blas && \
    cd ../Lapack && \
        ./get.Lapack && \
    cd ../Mumps && \
        ./get.Mumps && \
# Get METIS
    cd ../Metis && \
        ./get.Metis && \
##################
# Compile ipoptr
##################
    cd ../../ && \
    mkdir build && \
    cd build && \
    ../configure -with-pic --prefix=/venv/ && \
    make -j3 && \
    make test && \
    make install && \
    cd ./Ipopt/src/Interfaces/.libs/ && \
    ln -s libipopt.so.1 libipopt.so.3

ENV LD_LIBRARY_PATH=/venv/lib:$LD_LIBRARY_PATH
ENV PATH=/venv/bin:$PATH
ENV PKG_CONFIG_PATH=/venv/lib/pkgconfig:$PKG_CONFIG_PATH

RUN /bin/bash -l -c "pip3 install Cython"
RUN /bin/bash -l -c "pip3 install six"

RUN /bin/bash -l -c "pip3 install git+https://github.com/mechmotum/cyipopt.git@v0.3.0"

# Install petsc
RUN git clone -b release https://gitlab.com/petsc/petsc petsc \
    && cd petsc \
    && ./configure --COPTFLAGS="-O2" \
        --CXXOPTFLAGS="-O2" \
        --FOPTFLAGS="-O2" \
        --with-fortran-bindings=no \
        --with-debugging=0 \
        --download-blacs \
        --download-hypre \
        --download-metis \
        --download-mumps \
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu \
        --prefix=/venv/petsc \
    && make \
    && make install \
    && rm -rf /tmp/* \
    && rm -rf /petsc

ENV PETSC_DIR=/venv/petsc

# Install slepsc
RUN git clone -b release https://gitlab.com/slepc/slepc slepc \
    && cd slepc \
    && ./configure --prefix=/venv/slepc \
    && make SLEPC_DIR=$(pwd) \
    && make install SLEPC_DIR=$(pwd) \
    && rm -rf /tmp/* \
    && rm -rf /slepc

ENV SLEPC_DIR=/venv/slepc

RUN python -m pip install wheel cmake --no-cache-dir \
    && python -m pip install dev-fenics-ffc mpi4py petsc4py pybind11 --no-cache-dir \
    && python -m pip install slepc4py --no-cache-dir \
    && python -m pip install h5py --no-binary=h5py

# Install dolfin
RUN git clone https://bitbucket.org/JohannesHaubner/dolfin.git dolfin \
    && cd dolfin \
    && cmake -DCMAKE_INSTALL_PREFIX=/venv -DCMAKE_PREFIX_PATH=/venv -DCMAKE_BUILD_TYPE=Release -B build -S . \
    && cmake --build build -j6 \
    && cmake --install build

# This is basically what is in the /venv/share/dolfin/dolfin.conf
ENV MANPATH=/venv/share/man:$MANPATH
ENV CMAKE_PREFIX_PATH=/venv/lib/python3.10/site-packages:$CMAKE_PREFIX_PATH

# Install the python API
RUN cd dolfin/python \
    && python -m pip install . --no-cache-dir \
    && rm -rf /dolfin


#Install dolfin-adjoint


RUN /bin/bash -l -c "pip3 install --no-cache git+https://github.com/dolfin-adjoint/pyadjoint.git@constant-adjfloat"
RUN /bin/bash -l -c "pip3 install matplotlib"





CMD ["bash"]
