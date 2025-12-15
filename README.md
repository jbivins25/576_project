This work is all original, created by me.

There are multiple things included here, such as the LIFT pipeline and the feature matching demo. 
For perception:
  Simply run python(3) fm.py in a cloned repo. It will automatically run.
For navigation:
  Multitude of steps:
    Use pointCloud.py as a library to generate SFC, Path, and Trajectories
    Use the Dockerfile to run any flights on the Crazyflie 2.1
    The pcSplicer.py is to split an initial LiDAR scan into x*y tiles for easier use generating the SFC from the pointCloud library
