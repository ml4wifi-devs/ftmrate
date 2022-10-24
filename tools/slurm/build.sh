#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"

cd $NS3_DIR
./ns3 configure --build-profile=optimized
./ns3 build
