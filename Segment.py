import numpy as np


class Segment:

    def __init__(self, t1, t2, t3, base):
        stiffness = np.array([t1.E, t2.E, t3.E])
        torsion = np.array([t1.G, t2.G, t3.G])
        curve_x = np.array([t1.U_x, t2.U_x, t3.U_x])
        curve_y = np.array([t1.U_y, t2.U_y, t3.U_y])

        d_tip = np.array([t1.L, t2.L, t3.L]) + base  # position of tip of the tubes
        d_c = d_tip - np.array([t1.L_c, t2.L_c, t3.L_c])  # position of the point where tube bending starts
        points = np.hstack((0, base, d_c, d_tip))
        index = np.argsort(points)
        segment_length = 1e-5 * np.floor(1e5 * np.diff(np.sort(points)))

        e = np.zeros((3, segment_length.size))
        g = np.zeros((3, segment_length.size))
        u_x = np.zeros((3, segment_length.size))
        u_y = np.zeros((3, segment_length.size))

        for i in range(0, 3):
            aa = np.where(index == i + 1)  # Find where tube begins
            a = aa[0]
            bb = np.where(index == i + 4)  # Find where tube curve starts
            b = bb[0]
            cc = np.where(index == i + 7)  # Find where tube ends
            c = cc[0]
            if segment_length[a] == 0:
                a += 1
            if segment_length[b] == 0:
                b += 1
            if segment_length[a] == 0:
                a += 1
            if c.item() <= segment_length.size - 1:
                if segment_length[c] == 0:
                    c += 1

            e[i, np.arange(a, c)] = stiffness[i]
            g[i, np.arange(a, c)] = torsion[i]
            u_x[i, np.arange(b, c)] = curve_x[i]
            u_y[i, np.arange(b, c)] = curve_y[i]

        # Getting rid of zero lengths
        length = segment_length[np.nonzero(segment_length)]
        ee = e[:, np.nonzero(segment_length)]
        gg = g[:, np.nonzero(segment_length)]
        uu_x = u_x[:, np.nonzero(segment_length)]
        uu_y = u_y[:, np.nonzero(segment_length)]

        length_sum = np.cumsum(length)
        self.S = length_sum[length_sum + min(base) > 0] + min(base)  # s is segmented abscissa of tube after template

        # Truncating matrices, removing elements that correspond to the tube before the template
        e_t = ee[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.EI = (e_t.T * np.array([t1.I, t2.I, t3.I])).T
        g_t = gg[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.GJ = (g_t.T * np.array([t1.J, t2.J, t3.J])).T
        self.U_x = uu_x[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.U_y = uu_y[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))



