import numpy as np


class Example():
    def __init__(self, n, D_zones, I_zones, U_zones, f, name):
        if len(D_zones) != n:
            raise ValueError('The dimension of D_zones is wrong.')
        if len(I_zones) != n:
            raise ValueError('The dimension of I_zones is wrong.')
        if len(U_zones) != n:
            raise ValueError('The dimension of U_zones is wrong.')
        if len(f) != n:
            raise ValueError('The dimension of f is wrong.')
        self.n = n  # number of variables
        self.D_zones = np.array(D_zones)  # local condition
        self.I_zones = np.array(I_zones)  # initial set
        self.U_zones = np.array(U_zones)  # unsafe set
        self.f = f  # differential equation
        self.name = name  # name or identifier


examples = {
    1: Example(
        n=7,
        D_zones=[[-2, 2]] * 7,
        I_zones=[[-1.01, -0.99]] * 7,
        U_zones=[[1.8, 2]] * 7,
        f=[
            lambda x: -0.4 * x[0] + 5 * x[2] * x[3],
            lambda x: 0.4 * x[0] - x[1],
            lambda x: x[1] - 5 * x[2] * x[3],
            lambda x: 5 * x[4] * x[5] - 5 * x[2] * x[3],
            lambda x: -5 * x[4] * x[5] + 5 * x[2] * x[3],
            lambda x: 0.5 * x[6] - 5 * x[4] * x[5],
            lambda x: -0.5 * x[6] + 5 * x[4] * x[5]
        ],
        name='test_7dim'
    ),
    2: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[0, 1], [1, 2]],
        U_zones=[[-2, -0.5], [-0.75, 0.75]],
        f=[
            lambda x: x[1] + 2 * x[0] * x[1],
            lambda x: -x[0] - x[1] ** 2 + 2 * x[0] ** 2
        ],
        name='barr_1'
    ),
    3: Example(
        n=4,
        D_zones=[[-2, 2]] * 4,
        I_zones=[[0.5, 1.5]] * 4,
        U_zones=[[-2.4, -1.6]] * 4,
        f=[lambda x: x[0],
           lambda x: x[1],
           lambda x: x[2],
           lambda x: - 3980 * x[3] - 4180 * x[2] - 2400 * x[1] - 576 * x[0]
           ],
        name='hi_ord_4'
    ),
    4: Example(
        n=2,
        D_zones=[[1, 5]] * 2,
        I_zones=[[4, 4.5], [0.9, 1.1]],
        U_zones=[[1, 2], [2, 3]],
        f=[lambda x: -5.5 * x[1] + x[1] * x[1],
           lambda x: 6 * x[0] - x[0] * x[0],
           ],
        name='emsoft_c1'
    ),
    5: Example(
        n=3,
        D_zones=[[-2, 2]] * 3,
        I_zones=[[-0.25, 0.75], [-0.25, 0.75], [-0.75, 0.25]],
        U_zones=[[1, 2], [-2, -1], [-2, -1]],
        f=[lambda x: -x[1],
           lambda x: -x[2],
           lambda x: -x[0] - 2 * x[1] - x[2] + x[0] * x[0] * x[0],
           ],
        name='emsoft_c2'
    ),
    6: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-1 / 5, 1 / 5], [3 / 10, 7 / 10]],
        U_zones=[[-2, -1], [-2, -1]],
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[0] * x[1] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c3'
    ),
    7: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-1, 0], [-1, 0]],
        U_zones=[[1, 2], [1, 2]],
        f=[lambda x: -1 + x[0] * x[0] + x[1] * x[1],
           lambda x: 5 * (-1 + x[0] * x[1])
           ],
        name='emsoft_c4'
    ),
    8: Example(
        n=2,
        D_zones=[[-3, 3]] * 2,
        I_zones=[[-1 / 5, 1 / 5], [-1 / 5, 1 / 5]],
        U_zones=[[2, 3], [2, 3]],
        f=[lambda x: x[0] - x[0] * x[0] * x[0] + x[1] - x[0] * x[1] * x[1],
           lambda x: -x[0] + x[1] - x[0] * x[0] * x[1] - x[1] * x[1] * x[1]
           ],
        name='emsoft_c5'
    ),
    9: Example(
        n=2,
        D_zones=[[-1, 1]] * 2,
        I_zones=[[-1 / 10, 1 / 10], [-1 / 10, 1 / 10]],
        U_zones=[[1 / 2, 1], [1 / 2, 1]],
        f=[lambda x: -2 * x[0] + x[0] * x[0] + x[1],
           lambda x: x[0] - 2 * x[1] + x[1] * x[1]
           ],
        name='emsoft_c6'
    ),
    10: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-3 / 2, -1 / 2], [-3 / 2, -1 / 2]],
        U_zones=[[-1 / 2, 1 / 2], [1 / 2, 3 / 2]],
        f=[lambda x: -x[0] + x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c7'
    ),
    11: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-1 / 4, 1 / 4], [3 / 4, 3 / 2]],
        U_zones=[[1, 2], [1, 2]],
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c8'
    ),
    12: Example(
        n=7,
        D_zones=[[-2, 2]] * 7,
        I_zones=[[-1.01, -0.99]] * 7,
        U_zones=[[1.8, 2]] * 7,
        f=[lambda x: -0.4 * x[0] + 5 * x[2] * x[3],
           lambda x: 0.4 * x[0] - x[1],
           lambda x: x[1] - 5 * x[2] * x[3],
           lambda x: 5 * x[4] * x[5] - 5 * x[2] * x[3],
           lambda x: -5 * x[4] * x[5] + 5 * x[2] * x[3],
           lambda x: 0.5 * x[6] - 5 * x[4] * x[5],
           lambda x: -0.5 * x[6] + 5 * x[4] * x[5]
           ],
        name='emsoft_c9'
    ),
    13: Example(
        n=9,
        D_zones=[[-2, 2]] * 9,
        I_zones=[[0.99, 1.01]] * 9,
        U_zones=[[1.8, 2]] * 9,
        f=[
            lambda x: 3 * x[2] - x[0] * x[5],
            lambda x: x[3] - x[1] * x[5],
            lambda x: x[0] * x[5] - 3 * x[2],
            lambda x: x[1] * x[5] - x[3],
            lambda x: 3 * x[2] + 5 * x[0] - x[4],
            lambda x: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
            lambda x: 5 * x[3] + x[1] - 0.5 * x[6],
            lambda x: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
            lambda x: 2 * x[5] * x[7] - x[8]
        ],
        name='emsoft_c10'
    ),
    14: Example(
        n=2,
        D_zones=[[-3.5, 2], [-2, 1]],
        I_zones=[[1, 2], [-0.5, 0.5]],
        U_zones=[[-1.4, -0.6]] * 2,
        f=[
            lambda x: x[1],
            lambda x: -x[0] - x[1] + 1 / 3.0 * x[0] ** 3
        ],
        name='barr_2'
    ),
    15: Example(
        n=6,
        D_zones=[[-2, 2]] * 6,
        I_zones=[[0.5, 1.5]] * 6,
        U_zones=[[-2, -1.6]] * 6,
        f=[
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[3],
            lambda x: x[4],
            lambda x: x[5],
            lambda x: - 800 * x[5] - 2273 * x[4] - 3980 * x[3] - 4180 * x[2] - 2400 * x[1] - 576 * x[0]
        ],
        name='hi_ord_6'
    ),
    16: Example(
        n=8,
        D_zones=[[-2, 2]] * 8,
        I_zones=[[0.5, 1.5]] * 8,
        U_zones=[[-2, -1.6]] * 8,
        f=[
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[3],
            lambda x: x[4],
            lambda x: x[5],
            lambda x: x[6],
            lambda x: x[7],
            lambda x: -20 * x[7] - 170 * x[6] - 800 * x[5] - 2273 * x[4] - 3980 * x[3] - 4180 * x[2] - 2400 * x[
                1] - 576 * x[0]
        ],
        name='hi_ord_8'
    ),
    17: Example(
        n=12,
        D_zones=[[-2, 2]] * 12,
        I_zones=[[-0.1, 0.1]] * 12,
        U_zones=[[0, 0.5]] * 3 + [[0.5, 1.5]] * 4 + [[-1.5, -0.5]] + [[0.5, 1.5]] * 2 + [[-1.5, -0.5]] + [[0.5, 1.5]],
        f=[
            lambda x: x[3],
            lambda x: x[4],
            lambda x: x[5],
            lambda x: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
            lambda x: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
            lambda x: -769.2308 * x[2] - 770.2301 * x[5],
            lambda x: x[9],
            lambda x: x[10],
            lambda x: x[11],
            lambda x: 9.81 * x[1],
            lambda x: -9.81 * x[0],
            lambda x: -16.3541 * x[11] - 15.3846 * x[8]
        ],
        name='emsoft_c11'
    ),
    18: Example(
        n=4,
        D_zones=[[-2, 2]] * 4,
        I_zones=[[0.5, 1.5]] * 4,
        U_zones=[[-1.5, -0.5]] * 4,
        f=[
            lambda x: -0.5 * x[0] * x[0] - 0.5 * x[1] * x[1] - 0.125 * x[2] * x[2] - 2 * x[1] * x[2] + 2 * x[3] * x[
                3] + 1,
            lambda x: -x[0] * x[1] - 1,
            lambda x: -x[0] * x[2],
            lambda x: -x[0] * x[3],
        ],
        name='Raychaudhuri'
    ),
    19: Example(
        n=6,
        D_zones=[[0, 10]] * 6,
        I_zones=[[3, 3.1]] * 6,
        U_zones=[[4, 4.1], [4.1, 4.2], [4.2, 4.3], [4.3, 4.4], [4.4, 4.5], [4.5, 4.6]],
        f=[
            lambda x: -x[0] ** 3 + 4 * x[1] ** 3 - 6 * x[2] * x[3],
            lambda x: -x[0] - x[1] + x[4] ** 3,
            lambda x: x[0] * x[3] - x[2] + x[3] * x[5],
            lambda x: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
            lambda x: -2 * x[1] ** 3 - x[4] + x[5],
            lambda x: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
        ],
        name='sixdim'
    ),
    20: Example(
        n=10,
        D_zones=[[-2, 2]] * 10,
        I_zones=[[-0.1, 0.1]] * 10,
        U_zones=[[0, 0.5], [0, 0.5], [0, 0.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [-1.5, -0.5], [0.5, 1.5],
                 [-1.5, -0.5]],
        f=[
            lambda x: x[3],
            lambda x: x[4],
            lambda x: x[5],
            lambda x: -7253.4927 * x[0] + 1936.3639 * x[9] - 1338.7624 * x[3] + 1333.3333 * x[7],
            lambda x: -1936.3639 * x[8] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
            lambda x: -769.2308 * x[2] - 770.2301 * x[5],
            lambda x: x[8],
            lambda x: x[9],
            lambda x: 9.81 * x[1],
            lambda x: -9.81 * x[0]
        ],
        name='dim_12_10'
    ),
    21: Example(
        n=8,
        D_zones=[[-2, 2]] * 8,
        I_zones=[[-0.1, 0.1]] * 8,
        U_zones=[[0, 0.5], [0, 0.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [-1.5, -0.5], [0.5, 1.5],
                 [-1.5, -0.5]],
        f=[
            lambda x: x[2],
            lambda x: x[3],
            lambda x: -7253.4927 * x[0] + 1936.3639 * x[7] - 1338.7624 * x[2] + 1333.3333 * x[5],
            lambda x: -1936.3639 * x[6] - 7253.4927 * x[1] - 1338.7624 * x[3] - 1333.3333 * x[4],
            lambda x: x[6],
            lambda x: x[7],
            lambda x: 9.81 * x[1],
            lambda x: -9.81 * x[0]
        ],
        name='dim_12_8'
    ),
    22: Example(
        n=6,
        D_zones=[[-2, 2]] * 6,
        I_zones=[[1, 2]] * 6,
        U_zones=[[-1, -0.5]] * 6,
        f=[
            lambda x: x[0] * x[2],
            lambda x: x[0] * x[4],
            lambda x: (x[3] - x[2]) * x[2] - 2 * x[4] * x[4],
            lambda x: -(x[3] - x[2]) ** 2 - x[0] * x[0] + x[5] * x[5],
            lambda x: x[1] * x[5] + (x[2] - x[3]) * x[4],
            lambda x: 2 * x[1] * x[4] - x[2] * x[5],
        ],
        name='meym'
    ),
    23: Example(
        n=3,
        D_zones=[[-0.3, 0.3]] * 3,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 2,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 2,
        f=[
            lambda x: (x[1] + x[2]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
        ],
        name='sin_3'
    ),
    24: Example(
        n=5,
        D_zones=[[-0.3, 0.3]] * 5,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 4,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 4,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
        ],
        name='sin_5'
    ),
    25: Example(
        n=7,
        D_zones=[[-0.3, 0.3]] * 7,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 6,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 6,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
        ],
        name='sin_7'
    ),
    26: Example(
        n=9,
        D_zones=[[-0.3, 0.3]] * 9,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 8,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 8,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
        ],
        name='sin_9'
    ),
    27: Example(
        n=11,
        D_zones=[[-0.3, 0.3]] * 11,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 10,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 10,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
        ],
        name='sin_11'
    ),
    28: Example(
        n=13,
        D_zones=[[-0.3, 0.3]] * 13,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 12,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 12,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6] + x[6] + x[7]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
            lambda x: x[12],
            lambda x: -10 * (x[11] - x[11] ** 3 / 6) - x[1],
        ],
        name='sin_13'
    ),
    29: Example(
        n=15,
        D_zones=[[-0.3, 0.3]] * 15,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 14,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 14,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6] + x[6] + x[7] + x[7] + x[
                8]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
            lambda x: x[12],
            lambda x: -10 * (x[11] - x[11] ** 3 / 6) - x[1],
            lambda x: x[14],
            lambda x: -10 * (x[13] - x[13] ** 3 / 6) - x[1],
        ],
        name='sin_15'
    ),
    30: Example(
        n=17,
        D_zones=[[-0.3, 0.3]] * 17,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 16,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 16,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6] + x[6] + x[7] + x[7] + x[8] +
                       x[8] + x[9]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
            lambda x: x[12],
            lambda x: -10 * (x[11] - x[11] ** 3 / 6) - x[1],
            lambda x: x[14],
            lambda x: -10 * (x[13] - x[13] ** 3 / 6) - x[1],
            lambda x: x[16],
            lambda x: -10 * (x[15] - x[15] ** 3 / 6) - x[1],
        ],
        name='sin_17'
    ),
    31: Example(
        n=19,
        D_zones=[[-0.3, 0.3]] * 19,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 18,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 18,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6] + x[6] + x[7] + x[7] + x[8] +
                       x[8] + x[9] + x[9] + x[10]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
            lambda x: x[12],
            lambda x: -10 * (x[11] - x[11] ** 3 / 6) - x[1],
            lambda x: x[14],
            lambda x: -10 * (x[13] - x[13] ** 3 / 6) - x[1],
            lambda x: x[16],
            lambda x: -10 * (x[15] - x[15] ** 3 / 6) - x[1],
            lambda x: x[18],
            lambda x: -10 * (x[17] - x[17] ** 3 / 6) - x[1],
        ],
        name='sin_19'
    ),
    32: Example(
        n=21,
        D_zones=[[-0.3, 0.3]] * 21,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 20,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 20,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3] + x[3] + x[4] + x[4] + x[5] + x[5] + x[6] + x[6] + x[7] + x[7] + x[8] +
                       x[8] + x[9] + x[9] + x[10] + x[10] + x[11]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
            lambda x: x[6],
            lambda x: -10 * (x[5] - x[5] ** 3 / 6) - x[1],
            lambda x: x[8],
            lambda x: -10 * (x[7] - x[7] ** 3 / 6) - x[1],
            lambda x: x[10],
            lambda x: -10 * (x[9] - x[9] ** 3 / 6) - x[1],
            lambda x: x[12],
            lambda x: -10 * (x[11] - x[11] ** 3 / 6) - x[1],
            lambda x: x[14],
            lambda x: -10 * (x[13] - x[13] ** 3 / 6) - x[1],
            lambda x: x[16],
            lambda x: -10 * (x[15] - x[15] ** 3 / 6) - x[1],
            lambda x: x[18],
            lambda x: -10 * (x[17] - x[17] ** 3 / 6) - x[1],
            lambda x: x[20],
            lambda x: -10 * (x[19] - x[19] ** 3 / 6) - x[1],
        ],
        name='sin_21'
    )
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))


if __name__ == '__main__':
    print(get_example_by_id(1))
