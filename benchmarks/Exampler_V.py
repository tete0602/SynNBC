import numpy as np


class Example():
    def __init__(self, n, D_zones, f, name):
        if len(D_zones) != n:
            raise ValueError('The dimension of D_zones is wrong.')
        if len(f) != n:
            raise ValueError('The dimension of f is wrong.')
        self.n = n  # number of variables
        self.D_zones = np.array(D_zones)  # local condition
        self.f = f  # differential equation
        self.name = name  # name or identifier


examples = {
    0: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0] + x[0] * x[1],
            lambda x: -x[1]
        ],
        name='nonpoly0'
    ),
    1: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0] + 2 * x[0] ** 2 * x[1],
            lambda x: -x[1]
        ],
        name='nonpoly1'

    ),
    2: Example(
        n=3,
        D_zones=[[-10, 10]] * 3,
        f=[
            lambda x: -x[0],
            lambda x: -2 * x[1] + 0.1 * x[0] * x[1] ** 2 + x[2],
            lambda x: -x[2] - 1.5 * x[1]
        ],
        name='nonpoly2'
    ),
    3: Example(
        n=4,
        D_zones=[[-2, 2]] * 4,
        f=[lambda x: x[0],
           lambda x: x[1],
           lambda x: x[2],
           lambda x: - 3980 * x[3] - 4180 * x[2] - 2400 * x[1] - 576 * x[0]
           ],
        name='hi_ord_4'
    ),
    4: Example(
        n=3,
        D_zones=[[-10, 10]] * 3,
        f=[
            lambda x: -3 * x[0] - 0.1 * x[0] * x[1] ** 3,
            lambda x: -x[1] + x[2],
            lambda x: -x[2]
        ],
        name='nonpoly3'
    ),
    5: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0],
            lambda x: -x[1]
        ],
        name='poly0'
    ),
    6: Example(
        n=2,
        D_zones=[[-1.5, 1.5]] * 2,
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[0] * x[1] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c3'
    ),
    7: Example(
        n=3,
        D_zones=[[-10, 10]] * 3,
        f=[
            lambda x: -x[0] ** 3 - x[0] * x[2] ** 2,
            lambda x: -x[1] - x[0] ** 2 * x[1],
            lambda x: -x[2] + 3 * x[0] ** 2 * x[2] - 3 * x[2]
        ],
        name='poly1'
    ),
    8: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0] ** 3 + x[1],
            lambda x: -x[0] - x[1],
        ],
        name='poly2'
    ),
    9: Example(
        n=2,
        D_zones=[[-1, 1]] * 2,
        f=[lambda x: -2 * x[0] + x[0] * x[0] + x[1],
           lambda x: x[0] - 2 * x[1] + x[1] * x[1]
           ],
        name='emsoft_c6'
    ),
    10: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        f=[lambda x: -x[0] + x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c7'
    ),
    11: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c8'
    ),
    12: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
        ],
        name='poly3'

    ),
    13: Example(
        n=2,
        D_zones=[[-10, 10]] * 2,
        f=[
            lambda x: -x[0] - 1.5 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 0.5 * x[0] ** 3 * x[1] ** 2
        ],
        name='poly4'
    ),
    14: Example(
        n=2,
        D_zones=[[-1.6, 1.6], [-2, 2]],
        f=[
            lambda x: x[1],
            lambda x: -x[0] - x[1] + 1 / 3.0 * x[0] ** 3
        ],
        name='barr_2'
    ),
    15: Example(
        n=4,
        D_zones=[[-10, 10]] * 4,
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
            lambda x: -x[2] ** 3 + x[3],
            lambda x: -x[2] - x[3],
        ],
        name='poly5'
    )
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError(f'The example {name} was not found.')


if __name__ == '__main__':
    print(get_example_by_id(1))
