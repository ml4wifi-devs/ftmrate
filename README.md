# FTMRate

FTMRate is a rate adaptation algorithm for IEEE 802.11 networks which uses FTM to improve the per-frame selection of modulation and coding schemes.

  

## Installation

### FTMRate Repository

1. Clone the repository:
	```
	git clone https://github.com/ml4wifi-devs/ftmrate.git
	```

2. Go to project root directory and install requirements:
	```
	pip install -e .
	```

3.  **Attention!** In order to enable GPU acceleration for jax, run this additional command (For more info visit official JAX [install guide](https://github.com/google/jax#pip-installation-gpu-cuda)):
	```
	pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	```

### ns3 network simulator
To fully benefit from FTMRate, a network simulator ns3 needs to be installed on your machine. You can learn [here](https://www.nsnam.org/wiki/Installation) how to install the ns3, but we recommend to install it by cloning their gitlab dev repository.
1. Clone ns3 repository:
	```
	git clone https://gitlab.com/nsnam/ns-3-dev.git
	```

2. Then change directory to newly created `ns-3-dev/` and build the ns-3:
	```
	cd $YOUR_NS3_PATH
	./ns3 configure --build-profile=optimized --enable-examples --enable-tests
	./ns3 build
	```
3. Once you have built ns-3 (with examples enabled), you can test if the installation was successfull by running an example simulation:
	```
	./ns3 run wifi-simple-adhoc
	```

### ns3-ai
The ns3-ai module interconnects the ns-3 and FTMRate (or other python-writen software) by transferring data through
the shared memory pool. The memory can be accessed by both sides thus making the connection. Read more on ns3-ai on the
[official repository](https://github.com/hust-diangroup/ns3-ai).  **Attention!** The ns3-ai (as of 25.10.2022) is not compatible with the ns-3.36 or later. We have forked and modified the official ns3-ai repository to make it compatible with the 3.36 version. In order to install our compatible version folow the steps below.
1.  Clone our ns3-ai fork into ns3 contrib directory
	```
	cd $YOUR_NS3_PATH/contrib/
	git clone https://github.com/m-wojnar/ns3-ai.git
	```

2. Go to ns3-ai directory and checkout to *ml4wifi* branch:
	```
	cd "$YOUR_NS3_PATH/contrib/ns3-ai/"
	git checkout ml4wifi
	```
3. Install ns3-ai python interface:
	```
	pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"
	```
4. Rebuild the ns3:
	```
	cd $YOUR_NS3_PATH
	./ns3 configure --build-profile=optimized --enable-examples --enable-tests
	./ns3 build
	```

## Reproducing results

### Using Slurm workload manager
In order to produce a reliable results, hundreds of independant simulations needs to be run. [Slurm](https://slurm.schedmd.com/documentation.html) is a tool that we used to manage running multiple simulations on GPU simultaneously. We have collected all the Slurm using scripts in the `ftmrate/tools/slurm/` directory.  
In order to collect results from multiple WiFi scenarios to reproduce our results presented in the [article](LINK_TO_OUR_ARTICLE), you simply need to run
```
sbatch ftmrate/tools/slurm/run-ml-scenarios.sh
```
 to collect results into CSV format,  and
 ```
sbatch ftmrate/tools/generate-plots.sh
```
to aggregate results into matplotlip plots.

### Without Slurm
**TODO**
