# Hardware experiments

The following document provides a detailed description of the steps necessary to set up FTMRate on hardware. 
After going through all the steps, the hardware should be ready to run experiments.

**Note:** we performed experiments on [Intel Joule 570x](https://www.intel.com/content/www/us/en/products/sku/96414/intel-joule-570x-developer-kit/specifications.html)
boards with an Intel 8260 NIC. The configuration below has been adopted for this board with modified firmware that supports FTM.

**Note:** In some conditions, it may be necessary to use attenuators to limit the Wi-Fi range and allow measurements to 
be taken at shorter distances.

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/ml4wifi-devs/ftmrate_internal.git
	```

2. Go to the `experiments` directory and install requirements:
	```bash
	cd ftmrate_internal/experiments
	pip install -r requirements.txt
	```

Note that the packages listed in `requirements.txt` differ from the ones in the root directory. This file contains 
minimal requirements for running experiments on hardware. During the calibration process, you may need to install
matplotlib, pandas, and seaborn packages.

**Attention!** Before starting calibration and hardware tests, make sure that the appropriate MAC addresses of the 
station and access point are in the configuration files and scripts.

## Hyperparameters configuration

The `ftmrate.py` file contains three constants that need to be configured before running experiments based on the
hardware you are using:

- `FTM_INTERVAL` - interval of FTM measurements in seconds; the more frequent the measurements, the more dynamically 
  the station can move, but it generates more signaling overhead,
- `N_SAMPLES` - number of samples taken from the estimated distributions; the more samples, the more accurate the 
  estimation, but it requires more computation,
- `WIFI_MODES_RATES` - list of data rates for each MCS; the number, order and values of rates depend on the hardware 
  you are using.

## Calibration

Before running experiments, you need to calibrate the FTM measurements and FTMRate hyperparameters.
All needed files are in the `experiments/calibration` directory.

### FTM calibration

In the first place, it is worth to calibrate FTM. To do this, you need to run the `ftm.sh` script multiple times at 
different distances and save the results in a file. The script should run on the station.

```bash
cd $PATH_TO_FTMRATE_ROOT/experiments/calibration/ftm
sudo ./ftm.sh > d10_0.out
```

The file naming convention we use is `dX_Y.out`, where `X` is the distance and `Y` is the measurement number.
(it's worth to name files in this convention, because the scripts we use are adapted to this naming).

**Attention!** We recommend taking multiple measurements for each distance due to multipath fading, which can cause 
multiple measurements to fail and have a high variance. For each distance, we make 5 measurements arranged in a cross 
with measuring points 6 cm apart (i.e. a wavelength with a frequency of 2.4 GHz).

With all measurements saved, run a script that will parse the collected files and then run a script that will fit the 
line to the data and determine the correction factors for FTM:

```bash
python3 parse.py
python3 fit_ftm.py
```

The results (coefficients of the line) should be assigned to constants `FTM_COEFF` and `FTM_BIAS` in the `ftmrate.py` file.

### Kalman filter sensor noise calibration

Using the FTM measurements made earlier, you can calibrate the sensor noise for the Kalman filter. To do this, run the
`fit_kf.py` script (it requires fitted coefficients of the line):

```bash
cd $PATH_TO_FTMRATE_ROOT/experiments/calibration/ftm
python3 fit_kf.py --a <A_VAL> --b <B_VAL>
```

The results (variance of the sensor noise) should be assigned to constant `KF_SENSOR_NOISE` in the `ftmrate.py` file.

### Channel model calibration

The next step is to calibrate the channel model. To do this, you need to run the `rssi.sh` script multiple times at
different distances and save the results in a file. The script should run on the access point. While the script is 
running, the station should send frames to the AP so the AP can measure the RSSI (for example by running 
`send_frames_sta.py` script from the `experiments/calibration/success_probability` directory).

```bash
cd $PATH_TO_FTMRATE_ROOT/experiments/calibration/channel
sudo ./rssi.sh > d10_0.out
```

The file naming convention remains the same as for FTM calibration.

**Attention!** We recommend taking multiple measurements for each distance due to multipath fading, which can cause 
significant fluctuations in signal strength.

With all measurements saved, run a script that will parse the collected files and then run a script that will fit the
log-distance path loss model to the data and determine the exponent and shift:

```bash
python3 parse.py
python3 fit_channel.py
```

The results should be assigned to constants `RSSI_EXPONENT` and `RSSI_SHIFT` in the `ftmrate.py` file.

### Estimation of the success probability

The last step is to estimate the success probability of transmission for each MCS. The station should send a certain 
number of frames to the AP using specific MCS. At the same time, the AP should listen to the channel and count the 
number of frames it receives.

We have prepared a script that allows you to easily measure success probability for each MCS:

```bash
cd $PATH_TO_FTMRATE_ROOT/experiments/calibration/success_probability
sudo ./automate.py -d <DISTANCE> -m <MEASUREMENT NUMBER>
```

**Attention!** We recommend taking multiple measurements for each distance due to multipath fading, which can cause 
significant fluctuations in signal strength.

**Attention!** The script contains some constants that need to be configured before running experiments (e.g. 
IP addresses of the station and AP, password, directories for saving results, etc.).

Having all measurements saved, run a script that will gather all data and save it to a file (requires tshark to be installed):

```bash
cd $PATH_TO_FTMRATE_ROOT/experiments/calibration/success_probability
./parse.sh
```

Now you have to transform the distance to the RSSI (hence our method transforms the distance to the RSSI and then
transforms the RSSI to the expected rate). To do this, run the `covert_to_rssi.py` script with the estimated channel
model parameters:

```bash
python3 convert_to_rssi.py --rssi_exponent <EXPONENT> --rssi_shift <SHIFT>
```

At the end, run a script that will fit CDF of the normal distribution to the data for each MCS:

```bash
python3 fit_ps.py
```

Since successive measurements may have significantly different results (even for the same distance), it is recommended 
to view the results and manually discard outliers. The visualization is included in the `fit_ps.py` script.

The results should be assigned to array `WIFI_MODES_SNRS` in the `ftmrate.py` file.

## Experiments

We have prepared several scripts that allow you to run FTMRate on hardware. All necessary scripts are in the `experiments`
directory:

- `ftmrate.py` - contains constants and basic functions for running our algorithm,
- `kalman_filter.py` - a numpy implementation of one of the filtering algorithms used in FTMRate,
- `measure_distance.sh` - script for triggering FTM measurements,
- `conf` - configuration file for the `measure_distance.sh` script,
- `send_frames.py` - script for sending frames to the AP.

### Automation script

To conveniently carry out measurements, we have prepared a script that allows to automate the process of measuring
the performance of FTMRate. The script has the following parameters:

- `-r` or `--framerate` - the number of frames sent by scapy at once,
- `-d` or `--duration` - the duration of the experiment in seconds,
- `--useFtmrate` - flag indicating whether to use FTMRate or the default rate control algorithm.

The script connects to the AP and station via SSH, (optionally) runs FTMRate on the station, enables sending frames
from the station to the AP, and then runs tcpdump on the station and AP to collect the results. After the experiment,
it kills all processes on the station and AP. The script signals the beginning and end of the experiment with a sound 
signal.

**Attention!** The script contains some constants that need to be configured before running experiments (e.g. 
IP addresses of the station and AP, password, directories for saving results, etc.).

> Dealing with ' Blowfish has been deprecated' warning - [link](https://github.com/paramiko/paramiko/issues/2038#issuecomment-1117345478).