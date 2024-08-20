# TALocoMotion

This repository contains the code for training and testing different control paradigms on the unitreeGO2 robot. The optimization technique used is PPO (Proximal Policy Optimization), but other optimization techniques may also be implemented. The simulation enviroment used is MujocoX. 

The control paradigms currently implemented are:
- Position-based control
- Torque-based control
- Position + stiffness (and damping) control

The focus of this project is on the "position + stiffness (and damping)" control paradigm, as it is the main topic of research for the master's thesis.


## Requirements

This repo should be executable on both the EDA server as on a host machine itself. For execution on an abritrary host machine please use the "CPU" branch, this sets up everthing to work on the CPU only.

## Installation 
```
git clone git@github.com:gautica/TALocoMotion.git
conda create -n mjx python=3.12
conda activate mjx
cd TALocoMotion
pip install -r requirements.txt
```

## Setup EDA Server
In order to develop and execute the code on the server first install VSCode 1.88 on your machine. Then install the extensions: 
- Remote-SSH(v0.100.0)
- Remote Explorer (v0.4.0)
- Remote - Tunnels(v1.0.0)

Then you should be able to connect to the server via the controll panel using the command `Remote-SSH: Connect to Host...`.  Then add a new host by `Add New SSH Host...`. Now tipe in: 

```
ssh -X dspoljaric@eda01
```
The `-X` option is important as it enables X forwarding, which makes it possible to see the MujocoX display window on the actual machine.

After this is done the following procedure for setting up the conda enviroment is the same for developing on machine or on server:


## Training
Brax is allocating 75% of the GPU memory by default therefore it is necessary to tell it to use much less.
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python train.py
```
## Testing
In order to evaluate the model you have to first specify the path in `test.yaml`. This path should be the relative path from the folder TALocoMotion.
The model will then be tested with:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python test.py
```
Additionally to the previewed performance the tracking error and the foot z position are tracked and portraied in box plots and time dependent graphs.
These graphs are then found in the folder: `/outputs/graphs`.

## Generating terrains
With the recent update of MJX it is possible to load height fields. To generate those height field you can just import an greyscaled `.png` file. 
To generate terrains for the environment simply call.  
```
python utils/hm_gen.py
```
In the main function four different terrains are implied into one environment.

### Common MESA-LOADER Error
Somehow this error keeps happening, especially after restarting/suspending the computer. It was resolved after this blog post: https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris

To be exact it was resolved with this command: `conda install -c conda-forge libstdcxx-ng`