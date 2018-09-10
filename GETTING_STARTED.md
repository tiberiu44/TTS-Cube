# Quick-start installation

This installation guide applies to Linux and OSX. For Windows, you need to select a different way of installing DyNET.

For clarity, we will assume that you are use the user's home directory for installation. For a different location, just adapt the paths used in the following tutorial.

Also, this tutorial is intended for Python3. Make sure you have installed the following OS dependent tools:

- `Python3`, and `PIP3`
- `build-essentials` or equivalent (OSX should have everything up and running if you install `brew`)

**Step 1:** Clone this repository

```bash
git clone https://github.com/tiberiu44/TTS-Cube.git
```

**Step 2:** Install standard prerequisites

```bash
cd TTS-Cube
pip3 install -r requirements.txt
```

**Step 3:** Install DyNet with GPU support

Note: if you encounter any issues with this tutorial, please check the installation guidelines for [DyNet](https://github.com/clab/dynet)

You could always do a `pip3 install dynet`. However, this will install the CPU-only single threaded version of DyNet. TTS-Cube requires lots of computational power. As such, this tutorial will show how to install DyNet with GPU (CUDA) and Intel MKL support. If you use the default package, you will probably synthesize one second of audio in a couple of hours and you won't be able to train a model in a life-time.

First you need to download and install three external packages:

- NVIDIA CUDA SDK, available [here](https://developer.nvidia.com/cuda-downloads)
- NVIDIA CUDNN, available [here](https://developer.nvidia.com/cudnn) - in most cases you need to copy the CUDNN installation over the CUDA SDK installation (usually `/usr/local/cuda/`)
- INTEL MKL, avaialble [here](https://software.intel.com/en-us/mkl) - this should be optional since you are using CUDA, but from what I can tell, it helps to also compile MKL support into DyNet

Now, to compile DyNet with CUDA and MKL support, just follow the steps below:

```bash
pip install cython
cd ~
mkdir dynet-base
cd dynet-base

git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r 2355b22  # -r NUM specified a known working revision

cd dynet
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DMKL_ROOT=/opt/intel/mkl -DPYTHON=`which python3` -DBACKEND=cuda

make -j 2 # replace 2 with the number of available cores
make install

cd python
python3 ../../setup.py build --build-dir=.. --skip-build install
```

In the above commands we asume that you are using the default installation parameters for all external dependencies (CUDA, CUDNN, MKL). 