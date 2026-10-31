Claude refactored the code from the testing branch to submodules.
I pasted the main.py from testing branch asked it to create submodules.

Tuning steps:

Environement: no distubance and no obstacles
1.    tune ekf (get anees as close to 3 and nees inside more than 90)
2.    tune LQI (minimise rms and mean true crosstrack)
3.    re-tune ekf
4.    Change environment

Environemnt: add the disturbance (-1.0 rads/s in left wheel) and 2 walls
1.    tune LQI (primariily changing the integrator gain)
2.    tune EKF