# FTMRate

FTMRate is a rate adaptation algorithm for IEEE 802.11 networks which uses FTM to improve the per-frame selection of modulation and coding schemes.

## Installation

### FTMRate Repository

1. Clone the repository:
	```
	git clone https://github.com/ml4wifi-devs/ftmrate_internal.git
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

To fully benefit from FTMRate, the wifi-ftm-ns3 extension of the ns-3 network simulator needs to be installed on your machine. We show you how to install the ns-3 by cloning the official [wifi-ftm-ns3](https://github.com/tkn-tub/wifi-ftm-ns3) repository and integrate it with our FTMRate solution. You can read more on ns-3 installation process in the
[official installation notes](https://www.nsnam.org/wiki/Installation).

1. Clone the wifi-ftm-ns3 repository:
	```
	git clone git@github.com:tkn-tub/wifi-ftm-ns3.git
	```
2. Copy FTMRate contrib modules and simulation scenarios to the ns-3.33 directory of wifi-ftm-ns3:
	```
	cp -r $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/contrib/* $YOUR_NS3_PATH/contrib/
	cp $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/scratch/* $YOUR_NS3_PATH/scratch/
	```
3. Build ns-3:
	```
	cd $YOUR_NS3_PATH
	./waf configure -d optimized --enable-examples --enable-tests --disable-werror
	./waf
	```
4. Once you have built ns-3 (with examples enabled), you can test if the installation was successfull by running an example simulation:
	```
	./waf --run "wifi-simple-adhoc"
	```

### FTMRate and ns-3 synchronization (optional)

To flawlessly synchronize files between the FTMRate repository and the ns-3 installation, you can create symbolic links to the corresponding folders.
**Attention!** backup of all files in the `contrib` and `scratch` directories as creating symbolic links will require deleting these folders!

1. Remove `contrib` and `scratch` folders:
	```
    cd $YOUR_NS3_PATH
    rm -rf contrib
    rm -rf scratch
    ```
 
2. Create symbolic links:
    ```
    ln -s $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/contrib contrib
    ln -s $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/scratch scratch
    ```
   
3. Clone ns3-ai fork into ns-3's `contrib` directory - see [next section](#ns3-ai).

### ns3-ai

The ns3-ai module interconnects ns-3 and FTMRate (or any other python-writen software) by transferring data through a shared memory pool. 
The memory can be accessed by both sides, thus making the connection. Read more about ns3-ai at the
[official repository](https://github.com/hust-diangroup/ns3-ai).

1.  Clone ns3-ai into ns-3's `contrib` directory
	```
	cd $YOUR_NS3_PATH/contrib/
	git clone git@github.com:hust-diangroup/ns3-ai.git
	```
2. Install the ns3-ai python interface:
	```
	cd "$YOUR_NS3_PATH/contrib/ns3-ai/"
	pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"
	```
3. Rebuild ns-3:
	```
	cd $YOUR_NS3_PATH
	./waf configure -d optimized --enable-examples --enable-tests --disable-werror
	./waf
	```

## Reproducing results

We provide two ways of generating article results. One requires the Slurm workload manager to parallelize and accelerate this process. The other does not, but we treat this option as a proof of concept. To reproduce plots from the article with the `generate-plots.sh` script, you need a working TeX installation on your machine. To read how to enable LaTeX rendering in matplotlib see 
[this guide](https://matplotlib.org/stable/tutorials/text/usetex.html).

### Using the Slurm workload manager

To produce reliable results, many independent simulations need to be run. [Slurm](https://slurm.schedmd.com/documentation.html) is a tool that we used to manage running multiple simulations on a GPU simultaneously. We have collected all the Slurm scripts in the `ftmrate_internal/tools/slurm/` directory.  
To collect results from multiple Wi-Fi scenarios so to reproduce our results presented in our [article](LINK_TO_OUR_ARTICLE), you need to run
```
sbatch ftmrate_internal/tools/slurm/run-classic-scenarios.sh
sbatch ftmrate_internal/tools/slurm/run-ml-scenarios.sh
```
to collect results into CSV format  and
```
sbatch ftmrate_internal/tools/generate-plots.sh
```
to aggregate results into matplotlib plots.

#### Tuning hardware

When using GPU in slurm, you need to empirically determine the optimal number of tasks running per node. We store this value in
`TASKS_PER_NODE` variable in *run-ml-scenarios.sh* file. If set too high - a lack of memory might be encountered, if set too low - the computation efficiency would be suboptimal. The variable value is passed directly to the `sbatch` command as the `--ntasks-per-node` parameter. While working with our machines, the optimal number turned out to be about 5, so we suggest to start searching from that value.

### Without slurm

In case you don't have access to a slurm-managed cluster, we provide a slurm-less option to run all the simulations locally. Note that it would take an enormous amount of computation time to gather statistically reliable results, hence this slurm-less option is recommended only for soft tests with appropriately adjusted simulation parameters (in `tools/slurm/run-ml-scenarios.sh` and `tools/slurm/run-classic-scenarios.sh`). Nevertheless, to reproduce our article results without slurm, do the following steps.

1. It is recommended to set these environmental variables, otherwise our scripts may not discover appropriate paths:
	```
	export TOOLS_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/tools
	export ML4WIFI_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/ml4wifi
	export NS3_DIR=$YOUR_NS3_PATH
	```
	You should also update your Python path
	```
	export PYTHONPATH=$PYTHONPATH:$YOUR_PATH_TO_FTMRATE_ROOT
	```
2. Run simulations with our substituted `sbatch` script:
	```
	cd $YOUR_PATH_TO_FTMRATE_ROOT
	./tools/extras/sbatch ./tools/slurm/run-ml-scenarios.sh
	./tools/extras/sbatch ./tools/slurm/run-classic-scenarios.sh
	```
3. Aggregate results:
	```
	tools/slurm/generate-plots.sh
	```

