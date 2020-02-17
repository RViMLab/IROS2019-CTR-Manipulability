# Concentric Tube Robot Model and Manipulability Index
Repo for modeling of a concentric tube robot with 3 tubes and estimating its manipulability indices. It implements the methodology presented in the following publication:

M. Khadem, L. Da Cruz and C. Bergeles, "Force/Velocity Manipulability Analysis for 3D Continuum Robots," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018, pp. 4920-4926.

doi: 10.1109/IROS.2018.8593874

URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593874&isnumber=8593358

If you enjoy this repository and use it, please cite our paper
```
@inproceedings{khadem2018force,
  title={Force/velocity manipulability analysis for 3d continuum robots},
  author={Khadem, Mohsen and Da Cruz, Lyndon and Bergeles, Christos},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4920--4926},
  year={2018},
  organization={IEEE}
}
```


Dependencies:numpy, scipy, mpl_toolkits, matplotlib.

The CTR_Model module includes functions for modeling a 3-tube CTR. It accepts tubes parameters, joint variables q, initial value of joints q_0, input force f, tolerance for solver Tol, and a desired method for solving equations (1 for using BVP and 2 for IVP). The Class depends on two other modules Segment.py and Tube.py. CTR_Model module includes several functions:

ode-solver(u_init): This function accepts initial curvatures of thew most inner tube (u_x,u_y,u_z) and initial twist curvature of other tubes (an array with 5 elements) as input and calculates the shape of the CTR assuming the problem is an initial value problem. The solution is not accurate but very fast.

minimize(u_init): This function accepts initial curvatures of most inner tube (u_x,u_y,u_z) and twist curvature of other tubes as input and calculates the shape of the CTR assuming the problem is a boundary value problem. The solution is very accurate but slower than ivp. The given inputs are used only as initial guess for estimating the curvatures and can be set to zero. The output of this function is the correct initial curvatures satisfying the bvp.

jac(u_init): This function accepts initial curvatures of the most inner tube (u_x,u_y,u_z) and the twist curvature of other tubes as inputs and calculates the robot jacobian.

comp(u_init): This function accepts initial curvatures of the most inner tube (u_x,u_y,u_z) and the twist curvature of other tubes as inputs and calculates the robot compliance matrix.

Example.py shows how the module can be used to find robot shape and its manipulability indices.

