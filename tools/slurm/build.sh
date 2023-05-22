#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"

cd $NS3_DIR
./waf configure -d optimized --disable-werror --disable-python
./waf
