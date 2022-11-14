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

3.  **Attention!** To enable GPU acceleration for jax, run this additional command (For more info, see the official JAX [installation guide](https://github.com/google/jax#pip-installation-gpu-cuda)):
	```
	pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	```

### ns-3 network simulator

To fully benefit from FTMRate, the ns-3 network simulator needs to be installed on your machine. You read how to install ns-3 in the [official installation notes](https://www.nsnam.org/wiki/Installation), but we recommend to install it by cloning a git repository:

1. Clone the ns-3-dev repository:
	```
	git clone https://gitlab.com/nsnam/ns-3-dev.git
	```
2. Change directory to the newly created `ns-3-dev/` and build ns-3:
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

The ns3-ai module interconnects ns-3 and FTMRate (or any other python-writen software) by transferring data through a shared memory pool. 
The memory can be accessed by both sides, thus making the connection. Read more about ns3-ai at the
[official repository](https://github.com/hust-diangroup/ns3-ai).  **Attention!** The ns3-ai (as of 2022-10-25) is not compatible with ns-3.36 or later. We have forked and modified the official ns3-ai repository to make it compatible with the 3.36 version. In order to install our compatible version folow the steps below.

1.  Clone our ns3-ai fork into ns-3's `contrib` directory
	```
	cd $YOUR_NS3_PATH/contrib/
	git clone https://github.com/m-wojnar/ns3-ai.git
	```

2. Go to ns3-ai directory and checkout the *ml4wifi* branch:
	```
	cd "$YOUR_NS3_PATH/contrib/ns3-ai/"
	git checkout ml4wifi
	```
3. Install the ns3-ai python interface:
	```
	pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"
	```
4. Rebuild ns-3:
	```
	cd $YOUR_NS3_PATH
	./ns3 configure --build-profile=optimized --enable-examples --enable-tests
	./ns3 build
	```

## Reproducing results

### Using the Slurm workload manager

To produce reliable results, many independant simulations need to be run. [Slurm](https://slurm.schedmd.com/documentation.html) is a tool that we used to manage running multiple simulations on a GPU simultaneously. We have collected all the Slurm scripts in the `ftmrate/tools/slurm/` directory.  
To collect results from multiple WiFi scenarios so to reproduce our results presented in our [article](LINK_TO_OUR_ARTICLE), you need to run
```
sbatch ftmrate/tools/slurm/run-classic-scenarios.sh
sbatch ftmrate/tools/slurm/run-ml-scenarios.sh
```
to collect results into CSV format  and
```
sbatch ftmrate/tools/generate-plots.sh
```
to aggregate results into matplotlib plots.

#### Tuning hardware

When using GPU in slurm, you need to empirically determine the optimal number of tasks running per node. We store this value in
`TASKS_PER_NODE` variable in *run-ml-scenarios.sh* file. If set to high - a lack of memory might be encountered, if set to low - the computation efficiency would be suboptimal. The variable value is passed directly to the `sbatch` command as the `--ntasks-per-node` parameter. While working with our machines, the optimal number turned out to be about 5, so we suggest to start searching from that value.

### Without Slurm

In case you don't have access to a Slurm-managed cluster, we have provided a slurm-less option to run all the simulations locally. Note that it would take an enormous amount of computation time to gather statistically reliable results, hence this slurm-less option is recommended only for soft tests with appropriately adjusted simulation parameters (in `tools/slurm/run-ml-scenarios.sh` and `tools/slurm/run-classic-scenarios.sh` files). Nevertheless, to reproduce our article results without Slurm, do the following steps.

1. It is recommended to set those environmental variables, otherwise our scripts may not discover appropriate paths:
	```
	export TOOLS_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/tools
	export ML4WIFI_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/ml4wifi
	export NS3_DIR=$YOUR_NS3_PATH
	```
	You should also update your Python path
	```
	export PYTHONPATH=$PYTHONPATH:$YOUR_PATH_TO_FTMRATE_ROOT
	```
2. Run simulations with our substituted *sbatch* script:
	```
	cd $YOUR_PATH_TO_FTMRATE_ROOT
	./tools/extras/sbatch ./tools/slurm/run-ml-scenarios.sh
	./tools/extras/sbatch ./tools/slurm/run-classic-scenarios.sh
	```
3. Agregate results:
	```
	tools/slurm/generate-plots.sh
	```

