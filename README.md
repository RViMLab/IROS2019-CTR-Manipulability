# Concentric-Tube-Robot-Model-and-Manipulability-
The code for modelling of a concentric tube robot and estimating its manipulability.
Based on the following paper:
M. Khadem, L. Da Cruz and C. Bergeles, "Force/Velocity Manipulability Analysis for 3D Continuum Robots," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018, pp. 4920-4926.
doi: 10.1109/IROS.2018.8593874
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593874&isnumber=8593358


The CTR_Model Class includes functions for modeling a 3-tube CTR. It accepts tubes parameters, joint variables q, initial value of joints q_0, input force f, tolerance for solver Tol, and a desired method for solving equations (1 for using BVP and 2 for IVP). The Class depends on two other modules Segment.py and Tube.py. CTR_Model class has several functions including:

ode-solver(u_init): This function accepts initial curvatures of most inner tube (u_x,u_y,u_z) and twist curvature of other tubes as input and calculates the shape of the CTR assuming the problem as an initial value problem. The solution is not accurate but fast.

minimize(u_init): This function accepts initial curvatures of most inner tube (u_x,u_y,u_z) and twist curvature of other tubes as input and calculates the shape of the CTR assuming the problem as an boundary value problem. The solution is very accurate but slow. The given inputs are used only as initial guess for estimating the curvatures.

jac(u_init): This function accepts initial curvatures of most inner tube (u_x,u_y,u_z) and twist curvature of other tubes as input and calculates the robot jacobian.

comp(u_init): This function accepts initial curvatures of most inner tube (u_x,u_y,u_z) and twist curvature of other tubes as input and calculates the robot compliance matrix.

Example.py code shows how the class can be used to find robot shape and its manipulability indices.

