from math import pi, pow
import numpy as np


# This class includes physical parameters of each tube and code for segmenting them
class Tube:
    def __init__(self, length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness,
                 x_curvature, y_curvature):
        self.L = length
        self.L_s = length - length_curved
        self.L_c = length_curved
        self.J = (pi * (pow(diameter_outer, 4) - pow(diameter_inner, 4))) / 32
        self.I = (pi * (pow(diameter_outer, 4) - pow(diameter_inner, 4))) / 64
        self.E = stiffness
        self.G = torsional_stiffness
        self.U_x = x_curvature
        self.U_y = y_curvature
