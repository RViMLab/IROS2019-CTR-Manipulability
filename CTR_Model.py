import numpy as np
from scipy.integrate import solve_ivp
from Segment import Segment
from Tube import Tube
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import optimize
import time


class CTR_Model:
    def __init__(self, tube1, tube2, tube3, f, q, q_0, Tol, method):  # method 1,2 ---> bvp and ivp
        self.accuracy = Tol
        self.eps = 1.e-4
        self.method = method  # method 1,2 ---> bvp and ivp
        self.tube1, self.tube2, self.tube3, self.q, self.q_0 = tube1, tube2, tube3, q.astype(float), q_0.astype(float)
        # position of tubes' base from template (i.e., s=0)
        self.beta = self.q[0:3] + self.q_0[0:3]
        self.f = f.astype(float)
        self.segment = Segment(self.tube1, self.tube2, self.tube3, self.beta)

        self.span = np.append([0], self.segment.S)
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.alpha_1_0 = self.q[3] + self.q_0[3]  # initial twist angle for tube 1
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # initial rotation matrix
        self.alpha_0 = q[3:].reshape(3, 1) + q_0[3:].reshape(3, 1) - self.alpha_1_0  # initial twist angle for all tubes
        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 3))
        self.alpha = np.empty((0, 3))

    def reset(self):
        self.beta = self.q[0:3] + self.q_0[0:3]
        self.segment = Segment(self.tube1, self.tube2, self.tube3, self.beta)

        self.span = np.append([0], self.segment.S)
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.alpha_1_0 = self.q[3] + self.q_0[3]  # initial twist angle for tube 1
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # initial rotation matrix
        self.alpha_0 = self.q[3:].reshape(3, 1) + self.q_0[3:].reshape(3,
                                                                       1) - self.alpha_1_0  # initial twist angle for all tubes
        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 3))
        self.alpha = np.empty((0, 3))

    # ordinary differential equations for CTR with 3 tubes
    def ode_eq(self, s, y, ux_0, uy_0, ei, gj, f):
        # 1st element of y is curvature along x for first tube,
        # 2nd element of y is curvature along y for first tube
        # next 3 elements of y are curvatures along z, e.g., y= [ u1_z  u2_z ... ]
        # next 3 elements of y are twist angles, alpha_i
        # last 12 elements are r (position) and R (orientations), respectively
        dydt = np.empty([20, 1])
        tet2 = y[6]
        tet3 = y[7]
        u2 = np.array(
            [[np.cos(tet2), -np.sin(tet2), 0], [np.sin(tet2), np.cos(tet2), 0],
             [0, 0, 1]]).transpose() @ np.array([[y[0]], [y[1]], [y[2]]]) + dydt[
                 6] * np.array([[0], [0], [1]])  # Vector of curvature of tube 2
        u3 = np.array(
            [[np.cos(tet3), -np.sin(tet3), 0], [np.sin(tet3), np.cos(tet3), 0],
             [0, 0, 1]]).transpose() @ np.array([[y[0]], [y[1]], [y[2]]]) + dydt[
                 7] * np.array([[0], [0], [1]])
        u = np.array([y[0], y[1], y[2], u2[0, 0], u2[1, 0], y[3], u3[0, 0], u3[1, 0], y[4]])

        # estimating twist curvature and twist angles
        for i in np.argwhere(gj != 0):
            dydt[2 + i] = ((ei[i]) / (gj[i])) * (u[i * 3] * uy_0[i] - u[i * 3 + 1] * ux_0[i])  # ui_z
            dydt[5 + i] = y[2 + i] - y[2]  # alpha_i

        # estimating curvature of first tube along x and y
        du = np.zeros((3, 1))
        du[0, 0] = dydt[5] * ei[0] * y[1] + dydt[6] * ei[1] * y[1] + dydt[7] * ei[2] * y[1] + uy_0[0] * ei[0] * u[
            2] * np.cos(
            y[5]) + uy_0[1] * ei[1] * u[5] * np.cos(y[6]) + uy_0[2] * ei[2] * u[8] * np.cos(y[7]) + ux_0[0] * ei[0] * u[
                       2] * np.sin(
            y[5]) + ux_0[1] * ei[1] * u[5] * np.sin(y[6]) + ux_0[2] * ei[2] * u[8] * np.sin(y[7]) - ei[0] * u[1] * u[
                       2] * np.cos(
            y[5]) - ei[1] * u[4] * u[5] * np.cos(y[6]) - ei[2] * u[7] * u[8] * np.cos(y[7]) + gj[0] * u[1] * u[
                       2] * np.cos(
            y[5]) + gj[1] * u[4] * u[5] * np.cos(y[6]) + gj[2] * u[7] * u[8] * np.cos(y[7]) - ei[0] * u[0] * u[
                       2] * np.sin(
            y[5]) - ei[1] * u[3] * u[5] * np.sin(y[6]) - ei[2] * u[6] * u[8] * np.sin(y[7]) + gj[0] * u[0] * u[
                       2] * np.sin(
            y[5]) + gj[1] * u[3] * u[5] * np.sin(y[6]) + gj[2] * u[6] * u[8] * np.sin(y[7])
        du[1, 0] = uy_0[0] * ei[0] * u[2] * np.sin(y[5]) - dydt[6] * ei[1] * y[0] - dydt[7] * ei[2] * y[0] - ux_0[0] * \
                   ei[0] * u[2] * np.cos(
            y[5]) - ux_0[1] * ei[1] * u[5] * np.cos(y[6]) - ux_0[2] * ei[2] * u[8] * np.cos(y[7]) - dydt[5] * ei[0] * y[
                       0] + uy_0[1] * ei[1] * u[5] * np.sin(
            y[6]) + uy_0[2] * ei[2] * u[8] * np.sin(y[7]) + ei[0] * u[0] * u[2] * np.cos(y[5]) + ei[1] * u[3] * u[
                       5] * np.cos(
            y[6]) + ei[2] * u[6] * u[8] * np.cos(y[7]) - gj[0] * u[0] * u[2] * np.cos(y[5]) - gj[1] * u[3] * u[
                       5] * np.cos(
            y[6]) - gj[2] * u[6] * u[8] * np.cos(y[7]) - ei[0] * u[1] * u[2] * np.sin(y[5]) - ei[1] * u[4] * u[
                       5] * np.sin(
            y[6]) - ei[2] * u[7] * u[8] * np.sin(y[7]) + gj[0] * u[1] * u[2] * np.sin(y[5]) + gj[1] * u[4] * u[
                       5] * np.sin(
            y[6]) + gj[2] * u[7] * u[8] * np.sin(y[7])

        R = np.array(
            [[y[11], y[12], y[13]], [y[14], y[15], y[16]], [y[17], y[18], y[19]]])  # rotation matrix of 1st tube
        K_inv = np.diag(np.array([1 / np.sum(ei), 1 / np.sum(ei), 1 / np.sum(gj)]))
        Du = -K_inv @ du - K_inv @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ R.transpose() @ f
        dydt[0] = Du[0, 0]
        dydt[1] = Du[1, 0]

        # estimating R and r
        u_hat = np.array([[0, -y[2], y[1]], [y[2], 0, -y[0]], [-y[1], y[0], 0]])
        e3 = np.array([[0.0], [0.0], [1.0]])
        dr = R @ e3
        dR = (R @ u_hat).ravel()

        dydt[8] = dr[0, 0]
        dydt[9] = dr[1, 0]
        dydt[10] = dr[2, 0]

        for k in range(3, 12):
            dydt[8 + k] = dR[k - 3]
        return dydt.ravel()

    def ode_solver(self, u_init):
        # or not
        u1_xy_0 = np.array([[0.0], [0.0]])
        if self.method == 1:  # if method=1 use BVP solver
            u1_xy_0[0, 0] = u_init[0]
            u1_xy_0[1, 0] = u_init[1]
        elif self.method == 2:  # if method=2 use IVP solver
            u1_xy_0[0, 0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                            (self.segment.EI[0, 0] * self.segment.U_x[0, 0] +
                             self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                             self.segment.EI[1, 0] *
                             self.segment.U_y[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                             self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.cos(- self.alpha_0[2, 0]) +
                             self.segment.EI[2, 0] *
                             self.segment.U_y[2, 0] * np.sin(- self.alpha_0[2, 0]))
            u1_xy_0[1, 0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                            (self.segment.EI[0, 0] * self.segment.U_y[0, 0] +
                             -self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                             self.segment.EI[1, 0] *
                             self.segment.U_y[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                             -self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.sin(- self.alpha_0[2, 0]) +
                             self.segment.EI[2, 0] *
                             self.segment.U_y[2, 0] * np.cos(-self.alpha_0[2, 0]))
            print("IVP Method Selected Warning: Inacurate Solution")
        else:
            print("unidentified method, BVP method is used")
        uz_0 = u_init[2:].reshape(3, 1)

        # reset initial parameters for ode solver
        self.reset()
        for seg in range(0, len(self.segment.S)):
            # Initial conditions: 3 initial curvature of tube 1, 3 initial twist for tube 2 and 3, 3 initial angle,
            # 3 initial position, 9 initial rotation matrix
            y_0 = np.vstack((u1_xy_0, uz_0, self.alpha_0, self.r_0, self.R_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq(s, y, self.segment.U_x[:, seg], self.segment.U_y[:, seg],
                                                   self.segment.EI[:, seg], self.segment.GJ[:, seg], self.f),
                          [self.span[seg], self.span[seg + 1]],
                          y_0, method='RK23', max_step=self.accuracy)
            self.Length = np.append(self.Length, s.t)
            ans = s.y.transpose()
            self.u_z = np.vstack((self.u_z, ans[:, (2, 3, 4)]))
            self.alpha = np.vstack((self.alpha, ans[:, (5, 6, 7)]))
            self.r = np.vstack((self.r, ans[:, (8, 9, 10)]))
            dtheta2 = ans[-1, 3] - ans[-1, 2]
            dtheta3 = ans[-1, 4] - ans[-1, 2]
            # new boundary conditions for next segment
            uz_0 = self.u_z[-1, :].reshape(3, 1)
            self.r_0 = self.r[-1, :].reshape(3, 1)
            self.R_0 = np.array(ans[-1, 11:]).reshape(9, 1)
            self.alpha_0 = self.alpha[-1, :].reshape(3, 1)
            u1 = ans[-1, (0, 1, 2)].reshape(3, 1)
            if seg < len(
                    self.segment.S) - 1:  # enforcing continuity of moment to estimate initial curvature for next
                # segment

                K1 = np.diag(np.array([self.segment.EI[0, seg], self.segment.EI[0, seg], self.segment.GJ[0, seg]]))
                K2 = np.diag(np.array([self.segment.EI[1, seg], self.segment.EI[1, seg], self.segment.GJ[1, seg]]))
                K3 = np.diag(np.array([self.segment.EI[2, seg], self.segment.EI[2, seg], self.segment.GJ[2, seg]]))
                U1 = np.array([self.segment.U_x[0, seg], self.segment.U_y[0, seg], 0]).reshape(3, 1)
                U2 = np.array([self.segment.U_x[1, seg], self.segment.U_y[1, seg], 0]).reshape(3, 1)
                U3 = np.array([self.segment.U_x[2, seg], self.segment.U_y[2, seg], 0]).reshape(3, 1)

                GJ = self.segment.GJ
                GJ[self.segment.EI[:, seg + 1] == 0] = 0
                K1_new = np.diag(
                    np.array([self.segment.EI[0, seg + 1], self.segment.EI[0, seg + 1], self.segment.GJ[0, seg + 1]]))
                K2_new = np.diag(
                    np.array([self.segment.EI[1, seg + 1], self.segment.EI[1, seg + 1], self.segment.GJ[1, seg + 1]]))
                K3_new = np.diag(
                    np.array([self.segment.EI[2, seg + 1], self.segment.EI[2, seg + 1], self.segment.GJ[2, seg + 1]]))
                U1_new = np.array([self.segment.U_x[0, seg + 1], self.segment.U_y[0, seg + 1], 0]).reshape(3, 1)
                U2_new = np.array([self.segment.U_x[1, seg + 1], self.segment.U_y[1, seg + 1], 0]).reshape(3, 1)
                U3_new = np.array([self.segment.U_x[2, seg + 1], self.segment.U_y[2, seg + 1], 0]).reshape(3, 1)

                R_theta2 = np.array(
                    [[np.cos(self.alpha_0[1, 0]), -np.sin(self.alpha_0[1, 0]), 0],
                     [np.sin(self.alpha_0[1, 0]), np.cos(self.alpha_0[1, 0]), 0],
                     [0, 0, 1]])
                R_theta3 = np.array(
                    [[np.cos(self.alpha_0[2, 0]), -np.sin(self.alpha_0[2, 0]), 0],
                     [np.sin(self.alpha_0[2, 0]), np.cos(self.alpha_0[2, 0]), 0],
                     [0, 0, 1]])
                e3 = np.array([0, 0, 1]).reshape(3, 1)
                u2 = R_theta2.transpose() @ u1 + dtheta2 * e3
                u3 = R_theta3.transpose() @ u1 + dtheta3 * e3
                K_inv_new = np.diag(
                    np.array(
                        [1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1] + self.segment.EI[2, seg + 1]),
                         1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1] + self.segment.EI[2, seg + 1]),
                         1 / (self.segment.GJ[0, seg + 1] + self.segment.GJ[1, seg + 1] + self.segment.GJ[
                             2, seg + 1])]))
                u1_new = K_inv_new @ (K1 @ (u1 - U1) + R_theta2 @ K2 @ (u2 - U2) + R_theta3 @ K3 @ (
                        u3 - U3) + K1_new @ U1_new + R_theta2 @ K2_new @ U2_new + R_theta3 @ K3_new @ U3_new
                                      - R_theta2 @ K2_new @ (dtheta2 * e3) - R_theta3 @ K3_new @ (
                                              dtheta3 * e3))
                u1_xy_0 = u1_new[0:2, 0].reshape(2, 1)

        Cost = np.array([u1[0, 0] - self.segment.U_x[0, -1], u1[1, 0] - self.segment.U_y[0, -1], u1[2, 0], 0.0,
                         0.0])  # cost function for bpv solver includes 5 values: 3 twist curvature for tip of the 3
        # tubes and end curvature of the tip of the robot

        # finding twist curvature at the tip of the tubes
        d_tip = np.array([self.tube1.L, self.tube2.L, self.tube3.L]) + self.beta
        for i in range(1, 3):
            index = np.argmin(abs(self.Length - d_tip[i]))
            Cost[i + 2] = self.u_z[index, i]
        return Cost

    # estimating Jacobian for solving the BVP problem
    def jac_curvature(self, u_init):
        jac_bvp = np.zeros((5, 5))
        u_init_perturb = u_init
        cost = self.ode_solver(u_init)
        for i in range(0, 6):
            u_init_perturb[i] = u_init_perturb[i] + self.eps
            cost_perturb = (self.ode_solver(u_init_perturb, 1) - cost) / self.eps
            jac_bvp[:, i] = cost_perturb.reshape(5, )
            u_init_perturb[i] = u_init_perturb[i]
        return jac_bvp

    # Estimating Jacobian with respect to inputs (tubes' rotation and translation)
    def jac(self, u_init):
        jac = np.zeros((3, 6))
        self.ode_solver(u_init)
        r = self.r[-1, :]
        for i in range(0, 6):
            self.q[i] = self.q[i] + self.eps
            self.ode_solver(u_init)
            r_perturb = (self.r[-1, :] - r) / self.eps
            jac[:, i] = r_perturb.reshape(3, )
            self.q[i] = self.q[i] - self.eps
        return jac

    # Estimating Compliance matrix
    def comp(self, u_init):
        comp = np.zeros((3, 3))
        self.ode_solver(u_init)
        r = self.r[-1, :]
        for i in range(0, 3):
            self.f[i] = self.f[i] + 0.05
            self.ode_solver(u_init)
            r_perturb = (self.r[-1, :] - r) / 0.05
            comp[:, i] = r_perturb.reshape(3, )
            self.f[i] = self.f[i] - 0.05
        return comp

    # Solving the BVP problem using built-in scipy minimize module
    def minimize(self, u_init):
        u0 = u_init
        u0[0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                (self.segment.EI[0, 0] * self.segment.U_x[0, 0] +
                 self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                 self.segment.EI[1, 0] *
                 self.segment.U_y[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                 self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.cos(- self.alpha_0[2, 0]) +
                 self.segment.EI[2, 0] *
                 self.segment.U_y[2, 0] * np.sin(- self.alpha_0[2, 0]))
        u0[1] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                (self.segment.EI[0, 0] * self.segment.U_y[0, 0] +
                 -self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                 self.segment.EI[1, 0] *
                 self.segment.U_y[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                 -self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.sin(- self.alpha_0[2, 0]) +
                 self.segment.EI[2, 0] *
                 self.segment.U_y[2, 0] * np.cos(-self.alpha_0[2, 0]))
        res = optimize.anderson(self.ode_solver, u0, f_tol=1e-3)
        # another option for scalar cost function is  is res = optimize.minimize(self.ode_solver, u0,
        # method='Powell', options={'gtol': 1e-3, 'maxiter': 1000})
        return res

    # Solving the BVP problem using Newton method
    def bvp_solver(self, u_init):
        u_init[0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                    (self.segment.EI[0, 0] * self.segment.U_x[0, 0] +
                     self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                     self.segment.EI[1, 0] *
                     self.segment.U_y[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                     self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.cos(- self.alpha_0[2, 0]) +
                     self.segment.EI[2, 0] *
                     self.segment.U_y[2, 0] * np.sin(- self.alpha_0[2, 0]))
        u_init[1] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                    (self.segment.EI[0, 0] * self.segment.U_y[0, 0] +
                     -self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.sin(- self.alpha_0[1, 0]) +
                     self.segment.EI[1, 0] *
                     self.segment.U_y[1, 0] * np.cos(- self.alpha_0[1, 0]) +
                     -self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.sin(- self.alpha_0[2, 0]) +
                     self.segment.EI[2, 0] *
                     self.segment.U_y[2, 0] * np.cos(-self.alpha_0[2, 0]))
        cost = self.ode_solver(u_init, 1)
        d_cost = np.zeros(5, )
        Kp = np.diag(np.array([0.1, 0.05, 0.05, 0.05, 0.05]))
        Kd = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001]))
        Tol = 1e-1
        while np.sum(cost ** 2) > Tol:
            Jac = self.jac_curvature(u_init)
            du = np.linalg.inv(Jac) @ (- Kp @ cost.reshape(5, 1) + Kd @ d_cost.reshape(5, 1))
            u_init = u_init + du.ravel()
            cost_new = self.ode_solver(u_init, 1)
            d_cost = (cost_new - cost)
            cost = cost_new
        return u_init, cost


def main():
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
    J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))  # estimate jacobian matrix

    # Use this command if you wish to use boundary value problem (bvp) solver (very accurate but slower)
    u_init = CTR.minimize(np.concatenate((u1_xy_0, uz_0), axis=None))
    C = CTR.comp(u_init)  # estimate compliance matrix
    J = CTR.jac(u_init)  # estimate jacobian matrix

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
    scale = 0.1  # scaling manipulability plot
    eig, eig_v = np.linalg.eig(CME)
    eig = np.sqrt(eig)
    X, Y, Z = zip(center, center, center)
    Vectors = np.array([scale * eig, scale * eig, scale * eig]).T @ eig_v
    p0 = np.array([[Vectors[0, 0], 0, 0], [Vectors[1, 0], 0, 0], [Vectors[2, 0], 0, 0]])
    U, V, W = zip(p0)
    ax.quiver(X, Y, Z, U, V, W, color='g', label='CME')
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
    ax.quiver(X, Y, Z, U, V, W, color='r', label='UME')
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
    ax.quiver(X, Y, Z, U, V, W, color='y', label='VME')
    p0 = np.array([[Vectors[0, 1], 0, 0], [Vectors[1, 1], 0, 0], [Vectors[2, 1], 0, 0]])
    U, V, W = zip(p0)
    ax.quiver(X, Y, Z, U, V, W, color='y')
    p0 = np.array([[Vectors[0, 2], 0, 0], [Vectors[1, 2], 0, 0], [Vectors[2, 2], 0, 0]])
    U, V, W = zip(p0)
    ax.quiver(X, Y, Z, U, V, W, color='y')

    # plot the robot shape
    ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b', label='CTR Robot')
    ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                      [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                      [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.grid(True)
    plt.legend()
    plt.show()

    #np.savetxt('/home/mohsen/git_ws/CTR_Control_Matlab/FileName.csv', CTR.r, delimiter=',')

    return


if __name__ == "__main__":
    main()
