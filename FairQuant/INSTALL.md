# FairQuant Installation

The only additional installation is for [OpenBLAS](http://www.openblas.net).
You can download the library by following the [OpenBLAS's Installation Guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide) or running the following commands:

```shell
# Download the tar file
wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.6/OpenBLAS-0.3.6.tar.gz
tar -xzf OpenBLAS-0.3.6.tar.gz

# Set up installation path
export INSTALL_PREFIX=$HOME/OpenBLAS # or wherever you want to install OpenBLAS
mkdir $INSTALL_PREFIX

# Install
cd OpenBLAS-0.3.6
make
make PREFIX=$INSTALL_PREFIX install
```

Once you check that OpenBLAS has been successfully installed in `$INSTALL_PREFIX`, you can remove the tar file and the OpenBLAS-0.3.6 folder.
