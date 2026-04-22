# Variable Stiffness Locomotion (varstif_locomotion)
This repository provides the codebase for training and testing policies presented in this [paper](https://arxiv.org/abs/2502.09436), which was published at the IFAC conference 2025. For deeper insights you can also read into my [Master thesis](https://dario-spoljaric.com/assets/download/Masterarbeit.pdf) 

The optimization technique used is **PPO (Proximal Policy Optimization)**. The simulation enviroment used is Mujoco-MJX. 
Different controls are implemented to offer a fair comparison:
- Position-based control
- Torque-based control
- Position + stiffness (and damping) control

The focus of this project is on the "position + stiffness (and damping)" control paradigm, as it is the main topic of research for the master's thesis.

## Requirements
For the efficient execution of this repo, a GPU is strongly required. All of this code was executed under: `Ubuntu 22.0.4` with a GPU: ` NVIDIA GeForce GTX 1060 6GB`. 
Under normal settings such as 4096 parallelised environments with flat floor training required about 3GB of VRAM. 


## Installation 
```
git clone git@github.com:gautica/TALocoMotion.git
conda create -n mjx python=3.12
conda activate mjx
cd TALocoMotion
pip install -r requirements.txt
```

## Training
Per default brax is allocating 75 % of the GPU memory. This might not be necessary and therefore this parameter can be passed as "XLA_PYTHON_CLIENT_MEM_FRACTION".
Training can be executed with this line of code:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python train.py
```
The training settings can be investigated in the folder `config/`. This folder holds two configuration files. `train.yaml`holds the settings for Training and `test.yaml` the settings inherited by the test script. The overall policy settings are shared in `config/policy/`.

| ![Forward Movement](./docs/epoch0.gif) | ![Sidewards Movement](./docs/epoch200.gif) | ![Rotate Movement](./docs/epoch2000.gif) |
|:--------------------------------------:|:------------------------------------------:|:------------------------------------:|
| Epoch 0                        | Epoch 200                         | Epoch 2000                      |


## Testing
In order to evaluate the model you have to first specify the path in `test.yaml`. This path should be the relative path from the folder TALocoMotion. Beaware that all the control settings should be set accordingly.

In here different tests are included to test the robustness of the model. We are not only calculating the achieved reward, but rather implemented more real life test scenarios.

The model will then be tested with:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python test.py
```
Additionally to the previewed performance the tracking error and the foo
t z position are tracked and portraied in box plots and time dependent graphs.
These graphs are then found in the folder: `/outputs/graphs`.
If videos should be recorded (this can also be configured) they will be stored in `/outputs/videos`. 
Within the test script there are different evaluations experiments configured prehand. These experiments are:
- `heading_directions`: Tasks the policy to follow a target velocity. Per default 8 different directions are targeted (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) and investigated. Metrics such es average power, cost of transport(CoT) and tracking error are recorded and stored for later evaluation.
![Heading directions](./docs/velocity_tracking.gif) 
- `test_force_push`: This experiment exposes the policy to force pushes applied to the trunk. The experiment setting is as follows:
    - Policy tasked to walk 0.3 m/s into forward direction.
    - A randomised force, within the xy plane is applied at the 3 second of the experiment.
    - If the robot falls or triggers some early termination the experiment is regarded a failure and if the robot manages to walk at the 5th second mark the policy the experiment is regarded a success.

    This is all done in a parallelised manner. So the num envs defines the number of experiments.
![force push](./docs/force_push.gif) 

- `test_xy_random`: In this experiment the policy is faced with randomised commands that are changed within a sample command interval. If the policy manages to track the target velocity and keeps the tracking error below some threshold the experiment returns a success. This experiment is also parrallelised so the num envs directly relates to the number of experiments.

- `auto`: The task specified as auto executes all test cases implemented so far. First the policy will be tasked with the heading direction experiment, then it will be challenged with the force push and finally the performance on random sampled commands will be measured. 

# Multi model evaluation
For a more automated way of testing multiple models and storing the results a script was written, which can be called as: 
```
python automatic_eval.py
```
In this script models will automatically be loaded and their configurations without the need to define the control parameters in the `test.yaml`. Furthermore, it is possible to define a range of dates and all models within this range will be evaluated on the task `auto`.

## Generating terrains
With the recent update of MJX it is possible to load height fields. To generate those height field you can just import an greyscaled `.png` file. 
To generate terrains for the environment simply call.  
```
python utils/hm_gen.py
```

Per default it will generate a multi terrain, which includes different terrain types as shown:

| Terrain | Description                         |
|---------|-------------------------------------|
| ![Terrain 1](./docs/terrain_1.png) | Pyramid (direct or inverse)        |
| ![Terrain 2](./docs/terrain_2.png) | Gaussian belled hill               |
| ![Terrain 3](./docs/terrain_3.png) | Unstructured terrain               |
| ![Terrain 4](./docs/terrain_4.png) | Stairs in checkerboard arrangement |


### Common MESA-LOADER Error
Somehow this error keeps happening, especially after restarting/suspending the computer. It was resolved after this blog post: https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris

To be exact it was resolved with this command: `conda install -c conda-forge libstdcxx-ng`


# Testing on real hardware

Once a solid model was trained it can be easily deployed using my repository: [Unitree Mujoco Repo](https://github.com/DarioRepoRuler/unitree_mujoco/tree/main).

![Hardware Demo](docs/hardware_demo.gif)


# Citation

If you used this work or found it helpful please cite us:
```
@misc{spoljaric2025variablestiffnessrobustlocomotion,
      title={Variable Stiffness for Robust Locomotion through Reinforcement Learning}, 
      author={Dario Spoljaric and Yashuai Yan and Dongheui Lee},
      year={2025},
      eprint={2502.09436},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.09436}, 
}
```