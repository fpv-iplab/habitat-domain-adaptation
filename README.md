# On Embodied Visual Navigation in Real Environments Through Habitat
This repository hosts the code related to the paper:

Marco Rosano, Antonino Furnari, Luigi Gulino, and Giovanni Maria Farinella, "On Embodied Visual Navigation in Real Environments Through Habitat". International Conference on Pattern Recognition (ICPR). 2020. [Download the paper](https://iplab.dmi.unict.it/EmbodiedVN/ICPR_2020_Marco_Rosano.pdf)

For more details please see the project web page at [https://iplab.dmi.unict.it/EmbodiedVN](https://iplab.dmi.unict.it/EmbodiedVN/).

## Overview
This code is built on top of the Habitat-api/Habitat-lab project. Please see the [Habitat project page](https://github.com/facebookresearch/habitat-lab) for more details.

This repository provides the following components:

1. The official PyTorch implementation of the proposed Domain Adaptation approach, incuding the generalized noise models to simulate the inaccuracy of real sensors and actuators;

2. the virtual 3D model of the proposed environment, acquired using the Matterport 3D scanner, and used to carry on all the experiments;

3. the real images of the proposed environment, labeled with their pose. The sparse 3D reconstruction was performed using the [COLMAP Structure from Motion tool](https://colmap.github.io/), then aligned with the Matterport virtual 3D map.

4. An integration with [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train and evaluate navigation models on Habitat with domain translated images.

5. The checkpoints of the best performing navigation model and of the CycleGAN sim2real domain adaptation.

## Installation

### Requirements

* Python >= 3.7, use version 3.7 to avoid possible issues.
* Other requirements will be installed via `pip` in the following steps.

### Steps

0. (Optional) Create an Anaconda environment and install all on it ( `conda create -n DA-habitat python=3.7` )

1. Install the customized Habitat-lab (this repo):
	```bash
	git clone https://github.com/rosanom/habitat-domain-adaptation.git
	cd habitat-domain-adaptation/
	pip install -r requirements.txt
	python setup.py develop --all # install habitat and habitat_baselines

	```
2. (Optional) Download the [test scenes data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip), as suggested in the [Habitat-lab repository](https://github.com/facebookresearch/habitat-lab), and extract the `data` folder in zip to `habitat_domain_adaptation/data/` where `habitat_domain_adaptation/` is the github repository folder. To verify that the tool was successfully installed, run  `python examples/benchmark.py`.

3. Download our dataset [from here](https://iplab.dmi.unict.it/EmbodiedVN/real_world_nav_data.zip), and extract it to `habitat_domain_adaptation/`. Inside the `data` folder you should see this structure:
	```bash
	datasets/pointnav/orangedev/v1/...
	real_images/orangedev/...
	scene_datasets/orangedev/...
	orangedev_checkpoints/...

	```

4. Move to the `habitat_domain_adaptation/` parent directory and CLONE the `habitat-sim` repository, following the [install from source](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md) instructions. (The development and testing was done on commit `bfbe9fc30a4e0751082824257d7200ad543e4c0e`, if the last version of the simulator will not work properly, please consider to checkout to this commit).

5. Copy the custom noise model files and paste them in the simulator directory. Specifically, copy the `habitat_domain_adaptation/habitat_sim_noise_model/__init__.py` file to `habitat-sim-path/habitat-sim/`, overwrite it.
Copy the files in `habitat_domain_adaptation/habitat_sim_noise_model/agent/controls/` to `habitat-sim-path/habitat-sim/agent/controls/`. Overwrite the `__init__.py` file.

6. Continue the [`habitat-sim` installation procedure](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md). Skip the conda environment creation in point 2. if the conda env. was already created (point 0. of this guide) or if conda is not used.

### Test

To verify that `habitat_domain_adaptation` and `Habitat-sim` with the custom noise model are working correctly, take a look at the `habitat_domain_adaptation/test/test_noisy_sensors.py`. You can just run it as is or play with the simulator parameters in the script.

## Data Structure

All data can be found inside the `habitat_domain_adaptation/data/` folder:
* the `datasets/pointnav/orangedev/v1/...` folder contains the generated train and validation navigation episodes files;
* the `real_images/orangedev/...` folder contains the real world images of the proposed environment and the `csv` file with their pose information (obtained with COLMAP);
* the `scene_datasets/orangedev/...` folder contains the 3D mesh of the proposed environment.
* `orangedev_checkpoints/` is the folder where the checkpoints are saved during training. Place the checkpoint file here if you want to restore the training process or evaluate the model. The system will load the most recent checkpoint file.

## Config Files

There are two configuration files:

`habitat_domain_adaptation/configs/tasks/pointnav_orangedev.yaml`

and 

`habitat_domain_adaptation/habitat_baselines/config/pointnav/ddppo_pointnav_orangedev.yaml`.

In the first file you can change the robot's properties, the sensors used by the agent, the amount of noise to be introduced in the sensors and in the actuators.
```bash
	...
	# noisy actions
	ACTION_SPACE_CONFIG: customrobotnoisy
	NOISE_MODEL:
	ROBOT: Universal
	CONTROLLER: Proportional
	NOISE_STD: 0.05 # in meters
	ROT_NOISE_STD: 5.0 # in degrees
	...

```
```bash
	  ...
	  # noisy loc. sensor
	  POINTGOAL_WITH_GPS_COMPASS_SENSOR:
	    GOAL_FORMAT: "POLAR"
	    DIMENSIONALITY: 2
	    GPS_NOISE_AMOUNT: 0.2 # meters
	    ROT_NOISE_AMOUNT: 7.0 # degrees
	    GOAL_SENSOR_UUID: pointgoal_with_gps_compass
	  ...
```
In the second file you can change the learning parameters, if training or evaluating using real images, if using the CycleGAN sim2real model.
```bash
	  ...
	  TRAIN_W_REAL_IMAGES: True
	  EVAL_W_REAL_IMAGES: True
	  SIM_2_REAL: False #use cycleGAN
	  ...
```

## CycleGAN Integration

In order to use CycleGAN on Habitat for the sim2real domain adaptation, follow these steps:

1. clone the [CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) in the repository root (`habitat_domain_adaptation/`);
2. rename the `pytorch-CycleGAN-and-pix2pix` folder to `cyclegan`;
3. download our CycleGAN checkpoint file [from here](https://iplab.dmi.unict.it/EmbodiedVN/cyclegan_ckpt.zip) and extract it to `cyclegan/checkpoints/orangedev/`;
4. add the CycleGAN repo path to the `~/.bashrc` file. Open it with your text editor and add this line at the end:

	`export PYTHONPATH=$PYTHONPATH:/absolute/path/to/cyclegan/`

	then, source the `~/.bashrc` file.


## Train and Evaluation

To train the navigation model using the DD-PPO RL algorithm, run:

`sh habitat_baselines/rl/ddppo/single_node_orangedev.sh`

To evaluate the navigation model using the DD-PPO RL algorithm, run:

`sh habitat_baselines/rl/ddppo/single_node_orangedev_eval.sh`

For more information about DD-PPO RL algorithm, please check out the [habitat-lab dd-ppo repo page](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo).

## License
The code in this repository, the 3D models and the images of the proposed environment are MIT licensed. See the [LICENSE file](LICENSE) for details.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.
- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

## Citation
If you use the code/data of this repository in your research, please cite the paper:

```
@inproceedings{rosano2020navigation,
  title={On Embodied Visual Navigation in Real Environments Through Habitat},
  author={Rosano, Marco and Furnari, Antonino and Gulino, Luigi and Farinella, Giovanni Maria},
  booktitle={International Conference on Pattern Recognition (ICPR)}
  year={2020}
}
```
