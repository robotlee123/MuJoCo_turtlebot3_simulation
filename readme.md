# 41014 42043 TurtleBot3 simulation

## Explanation
Simulation of turtlebot3 in MuJoCo simulator. The robot must navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked goal.

## Environment
Tested on:
Windows 11
python 3.12.10
Nvdia GPU 

## Group information
Group number: Project 31
Group composition: Edmond Lee (26300792)

## Install info
Define here the dependencies for your project
```bash
pip install -r requirements.txt
```

## Run commands
To run the simulation use:

cd to the MuJoCo_turtlebot3_simulation directory
```bash
cd MuJoCo_turtlebot3_simulation
```
Run the main.py
```bash
python main.py
```
The mujoco viewer will appear with the simulated world.

To see the camera feed from the robot change the 'draw' variable to true in main.py
