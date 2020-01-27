
import numpy as np
from Tube import Tube
from CTR_Model import CTR_Model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Defining parameters of each tube, numbering starts with the most inner tube
# length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 6.43e+10, 2.50e+10, 21.3, 0)
tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 5.25e+10, 2.14e+10, 13.108, 0)
tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 4.71e+10, 2.97e+10, 3.5, 0)
# Joint variables
q = np.array([0.01, 0.015, 0.019, np.pi / 2, 5 * np.pi / 2, 3 * np.pi / 2])
# Initial position of joints
q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
# initial twist (for ivp solver)
uz_0 = np.array([0.0, 0.0, 0.0])
u1_xy_0 = np.array([[0.0], [0.0]])
# force on robot tip along x, y, and z direction
f = np.array([0, 0, 0]).reshape(3, 1)

# Use this command if you wish to use initial value problem (ivp) solver (less accurate but faster)
CTR = CTR_Model(tube1, tube2, tube3, f, q, q_0, 0.01, 1)
C = CTR.comp(np.concatenate((u1_xy_0, uz_0), axis=None))  # estimate compliance matrix
J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))   # estimate jacobian matrix

# Use this command if you wish to use boundary value problem (bvp) solver (very accurate but slower)
u_init = CTR.minimize(np.concatenate((u1_xy_0, uz_0), axis=None))
C = CTR.comp(u_init)  # estimate compliance matrix
J = CTR.jac(u_init)   # estimate jacobian matrix


# Plotting the robot and principal axes of manipulability ellipsoids
print("--- %s seconds ---" % (time.time() - start_time))
fig = plt.figure()
ax = plt.axes(projection='3d')
center = CTR.r[-1, :]
# Stiffness matrix
Lambda = np.diag(np.array([1, 1, 1]))
VME = J[:, 0:4] @ J[:, 0:4].T  # Velocity manipulability
CME = C @ C.T  # Compliance manipulability
UME = C.T @ Lambda.T @ J @ J.T @ Lambda @ C  # Unified Force-Velocity manipulability

# plotting compliance manipulability in green
scale = 0.1 # scaling manipulability plot
eig, eig_v = np.linalg.eig(CME)
eig = np.sqrt(eig)
X, Y, Z = zip(center, center, center)
Vectors = np.array([scale * eig, scale * eig, scale * eig]).T @ eig_v
p0 = np.array([[Vectors[0, 0], 0, 0], [Vectors[1, 0], 0, 0], [Vectors[2, 0], 0, 0]])
U, V, W = zip(p0)
print(C)
ax.quiver(X, Y, Z, U, V, W, color='g',label='CME')
p0 = np.array([[Vectors[0, 1], 0, 0], [Vectors[1, 1], 0, 0], [Vectors[2, 1], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='g')
p0 = np.array([[Vectors[0, 2], 0, 0], [Vectors[1, 2], 0, 0], [Vectors[2, 2], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='g')

# plotting force-velocity manipulability in red
scale = 1  # scaling manipulability plot

eig, eig_v = np.linalg.eig(UME)
eig = np.sqrt(eig)
X, Y, Z = zip(center, center, center)
Vectors = np.array([scale * eig, scale * eig, scale * eig]).T @ eig_v
p0 = np.array([[Vectors[0, 0], 0, 0], [Vectors[1, 0], 0, 0], [Vectors[2, 0], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='r',label='UME')
p0 = np.array([[Vectors[0, 1], 0, 0], [Vectors[1, 1], 0, 0], [Vectors[2, 1], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='r')
p0 = np.array([[Vectors[0, 2], 0, 0], [Vectors[1, 2], 0, 0], [Vectors[2, 2], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='r')

# plotting velocity manipulability in yellow
scale = 0.01  # scaling manipulability plot
eig, eig_v = np.linalg.eig(VME)
eig = np.sqrt(eig)
X, Y, Z = zip(center, center, center)
Vectors = np.array([scale * eig, scale * eig, scale * eig]).T @ eig_v
p0 = np.array([[Vectors[0, 0], 0, 0], [Vectors[1, 0], 0, 0], [Vectors[2, 0], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='y',label='VME')
p0 = np.array([[Vectors[0, 1], 0, 0], [Vectors[1, 1], 0, 0], [Vectors[2, 1], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='y')
p0 = np.array([[Vectors[0, 2], 0, 0], [Vectors[1, 2], 0, 0], [Vectors[2, 2], 0, 0]])
U, V, W = zip(p0)
ax.quiver(X, Y, Z, U, V, W, color='y')

# plot the robot shape
ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b',label='CTR Robot')
ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                  [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                  [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.grid(True)
plt.legend()
plt.show()