# FTMRate

[![DOI](https://img.shields.io/badge/DOI-10.1109/WoWMoM57956.2023.00039-blue.svg)](https://ieeexplore.ieee.org/document/10195443)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.7875867.svg)](https://doi.org/10.5281/zenodo.7875867)

FTMRate is a rate adaptation algorithm for IEEE 802.11 networks which uses the IEEE 802.11 fine timing measurement (FTM) protocol to improve the per-frame selection of modulation and coding schemes. Its detailed operation and a performance analysis can be found in:

- Wojciech Ciezobka, Maksymilian Wojnar, Katarzyna Kosek-Szott, Szymon Szott, and Krzysztof Rusek. "FTMRate: Collision-Immune Distance-based Data Rate Selection for IEEE 802.11 Networks." 24th IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM), 2023. [[preprint](https://arxiv.org/pdf/2304.10140.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/10195443)]
- Wojciech Ciezobka, Maksymilian Wojnar, Krzysztof Rusek, Katarzyna Kosek-Szott, Szymon Szott, Anatolij Zubow, and Falko Dressler. "Using Ranging for Collision-Immune IEEE 802.11 Rate Selection with Statistical Learning" (under review).

## Installation

**Note:** if you want to run FTMRate in an in-band scenario, please follow the instructions in the `wifi_ftm_ns3` branch.

### FTMRate Repository

1. Clone the repository:
	```
	git clone https://github.com/ml4wifi-devs/ftmrate.git
	```

2. Go to project root directory and install requirements:
	```
	cd ftmrate
	pip install -e .
	```

3.  **Attention!** To enable GPU acceleration for JAX, run this additional command (For more info, see the official JAX [installation guide](https://github.com/google/jax#installation)):
	```
	pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	```

### ns-3 network simulator

To fully benefit from FTMRate, the ns-3 network simulator needs to be installed on your machine. We show you how to install the ns-3 by downloading the official distribution and integrate it with our FTMRate solution. You can read more on ns-3 installation process in the
[official installation notes](https://www.nsnam.org/wiki/Installation).

1. Download and unzip ns-3.36.1:
	```
	wget https://www.nsnam.org/releases/ns-allinone-3.36.1.tar.bz2
	tar -xf ns-allinone-3.36.1.tar.bz2
	mv ns-allinone-3.36.1/ns-3.36.1 $YOUR_NS3_PATH
	```
2. Copy FTMRate contrib modules and simulation scenarios to the ns-3-dev directory:
	```
	cp -r $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/contrib/* $YOUR_NS3_PATH/contrib/
	cp $YOUR_PATH_TO_FTMRATE_ROOT/ns3_files/scratch/* $YOUR_NS3_PATH/scratch/
	```
3. Build ns-3:
	```
	cd $YOUR_NS3_PATH
	./ns3 configure --build-profile=optimized --enable-examples --enable-tests
	./ns3 build
	```
4. Once you have built ns-3 (with examples enabled), you can test if the installation was successful by running an example simulation:
	```
	./ns3 run wifi-simple-adhoc
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
[official repository](https://github.com/hust-diangroup/ns3-ai).  **Attention!** ns3-ai (as of 2022-10-25) is not compatible with the CMake based ns-3 versions. We have forked and modified the official ns3-ai repository to make it compatible with the 3.36.1 version. In order to install our compatible version follow the steps below.

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
3. Install the ns3-ai Python interface:
	```
	pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"
	```
4. Rebuild ns-3:
	```
	cd $YOUR_NS3_PATH
	./ns3 configure --build-profile=optimized --enable-examples --enable-tests
	./ns3 build
	```
 
Using our fork of ns3-ai, use can run ns-3 simulations in optimized or debug mode. To select mode change `debug` flag in ns3-ai `Experiment` declaration:
```python
exp = Experiment(mempool_key, mem_size, scenario, ns3_path, debug=False)
```

```python
exp = Experiment(mempool_key, mem_size, scenario, ns3_path, debug=True)
```

## Reproducing results

We provide two ways of generating article results. One requires the Slurm workload manager to parallelize and accelerate this process. The other does not, but we treat this option as a proof of concept. To reproduce plots from the article with the `generate-plots.sh` script, you need a working TeX installation on your machine. To read how to enable LaTeX rendering in matplotlib see 
[this guide](https://matplotlib.org/stable/tutorials/text/usetex.html).

It is recommended to set these environmental variables, otherwise our scripts may not discover appropriate paths:
```
export TOOLS_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/tools
export ML4WIFI_DIR=$YOUR_PATH_TO_FTMRATE_ROOT/ml4wifi
export NS3_DIR=$YOUR_NS3_PATH
```
You should also update your Python path
```
export PYTHONPATH=$PYTHONPATH:$YOUR_PATH_TO_FTMRATE_ROOT
```

### Using the Slurm workload manager

To produce reliable results, many independent simulations need to be run. [Slurm](https://slurm.schedmd.com/documentation.html) is a tool that we used to manage running multiple simulations on a GPU simultaneously. We have collected all the Slurm scripts in the `ftmrate/tools/slurm/` directory.  

#### Adapting scripts to your Slurm configuration (important!)

All scripts in the `ftmrate/tools/slurm/` directory are configured to work with our server configuration, so you need to adjust the scripts to work with your setup. The most important is to change the partition name and the software collection definition. The simplest way to do this is to:

1. Remove the `-p gpu` flag from the `sbatch` command in all Slurm scripts.
2. Replace the `#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l` line with `#!/bin/bash` in all Slurm scripts.

#### Running simulations

To collect results from multiple Wi-Fi scenarios so to reproduce our results presented in our article ([preprint](https://arxiv.org/pdf/2304.10140.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/10195443), [Zenodo](https://zenodo.org/records/7875867)), you need to run
```
sbatch ftmrate/tools/slurm/run-all-scenarios.sh
```
to collect results into CSV format  and
```
sbatch ftmrate/tools/generate-plots.sh
```
to aggregate results into matplotlib plots.

#### Tuning hardware

When using GPU in slurm, you need to empirically determine the optimal number of tasks running per node. We store this value in
`TASKS_PER_NODE` variable in *run-ml-scenarios.sh* file. If set too high - a lack of memory might be encountered, if set too low - the computation efficiency would be suboptimal. The variable value is passed directly to the `sbatch` command as the `--ntasks-per-node` parameter. While working with our machines, the optimal number turned out to be about 5, so we suggest to start searching from that value.

### Without slurm

In case you don't have access to a slurm-managed cluster, we provide a slurm-less option to run all the simulations locally. Note that it would take an enormous amount of computation time to gather statistically reliable results, hence this slurm-less option is recommended only for soft tests with appropriately adjusted simulation parameters (in `tools/slurm/run-ml-scenarios.sh` and `tools/slurm/run-classic-scenarios.sh`). Nevertheless, to reproduce our article results without slurm, run simulations with our substituted `sbatch` script:

```
cd $YOUR_PATH_TO_FTMRATE_ROOT
./tools/extras/sbatch ./tools/slurm/run-all-scenarios.sh
```

# How to reference FTMRate?

```
@INPROCEEDINGS{ciezobka2023ftmrate,
  author={Ciezobka, Wojciech and Wojnar, Maksymilian and Kosek-Szott, Katarzyna and Szott, Szymon and Rusek, Krzysztof},
  booktitle={2023 IEEE 24th International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM)}, 
  title={{FTMRate: Collision-Immune Distance-based Data Rate Selection for IEEE 802.11 Networks}}, 
  year={2023},
  volume={},
  number={},
  pages={242--251},
  doi={10.1109/WoWMoM57956.2023.00039}
}
```
