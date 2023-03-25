from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import sympy as sp
from benchmarks.Exampler_V import Example, get_example_by_id
import torch
import re
import gurobipy as gp


def split_bounds(bounds, n):
    """
    Divide an n-dimensional cuboid into 2^n small cuboids, and output the upper and lower bounds of each small cuboid.

    parameter: bounds: An array of shape (n, 2), representing the upper and lower bounds of each dimension of an
    n-dimensional cuboid.

    return:
        An array with a shape of (2^n, n, 2), representing the upper and lower bounds of the divided 2^n small cuboids.
    """

    if n == bounds.shape[0]:
        return bounds.reshape((-1, *bounds.shape))
    else:
        # Take the middle position of the upper and lower bounds of the current dimension as the split point,
        # and divide the cuboid into two small cuboids on the left and right.
        mid = (bounds[n, 0] + bounds[n, 1]) / 2
        left_bounds = bounds.copy()
        left_bounds[n, 1] = mid
        right_bounds = bounds.copy()
        right_bounds[n, 0] = mid
        # Recursively divide the left and right small cuboids.
        left_subbounds = split_bounds(left_bounds, n + 1)
        right_subbounds = split_bounds(right_bounds, n + 1)
        # Merge the upper and lower bounds of the left and right small cuboids into an array.
        subbounds = np.concatenate([left_subbounds, right_subbounds])
        return subbounds


class CounterExampleFinder():
    def __init__(self, example: Example, config):
        self.n = example.n
        self.D_zones = example.D_zones
        self.f = example.f
        self.choice = config.CHOICE
        self.eps = 0.05
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.split_D_zones = split_bounds(np.array(self.D_zones), 0)
        self.config = config

    def find_V(self, V, sample_nums=10):
        samples = []
        satisfy = True

        if isinstance(V, str):
            V = sp.sympify(V)
        if self.choice[0] == 0:
            opt = sp.lambdify(self.x, V)
            # Set an initial guess.
            x0 = np.zeros(shape=(self.n))
            res = minimize(fun=lambda x: opt(*x), x0=x0, bounds=self.D_zones)
            if res.fun < 0:
                print('Found counterexample V:', res.x)
                samples.append(res.x)
                samples.extend(self.generate_sample(res.x, sample_nums - 1))
                satisfy = False
        elif self.choice[0] == 1:
            V_str = str(V)
            res, x_values, status = self.get_min_value(V_str, self.D_zones)
            x_values = np.array(x_values)

            if res < 0 and status == True:
                print('Found counterexample V:', x_values)
                samples.append(x_values)
                samples.extend(self.generate_sample(x_values, sample_nums - 1))
                satisfy = False
        samples = np.array([x for x in samples if self.is_counter_example(V, x, 'V')])
        return samples, satisfy

    def find_diff(self, V, sample_nums=10):
        samples = []
        satisfy = True
        count = 0

        if isinstance(V, str):
            V = sp.sympify(V)

        if self.choice[1] == 0:
            opt = sp.lambdify(self.x, V)

            x = self.x
            DV = sum([sp.diff(V, x[i]) * self.f[i](x) for i in range(self.n)])
            optDV = sp.lambdify(x, DV)

            margin = 0.00
            constraint = NonlinearConstraint(lambda x: opt(*x), -margin, margin)
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones
            for bound in bounds:
                x0 = (bound.T[0] + bound.T[1]) / 2
                res = minimize(fun=lambda x: -optDV(*x), x0=x0, bounds=bound, constraints=constraint)
                if -res.fun < 0:
                    pass
                else:
                    samples.append(res.x)
                    samples.extend(self.generate_sample(res.x, sample_nums - 1))
                    satisfy = False
                    count += 1
        elif self.choice[1] == 1:
            V_str = str(V)
            x = self.x
            DV = sum([sp.diff(V, x[i]) * self.f[i](x) for i in range(self.n)])
            DV = sp.expand(DV)
            DV_str = str(DV)
            DV_str = '-(' + DV_str + ')'
            margin = 0.00
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones

            for bound in bounds:
                res, x_values, status = self.get_min_value(DV_str, bound, V_x=V_str, margin=margin)
                x_values = np.array(x_values)

                if -res < 0 or status == False:
                    pass
                else:
                    samples.append(x_values)
                    samples.extend(self.generate_sample(x_values, sample_nums - 1))
                    satisfy = False
                    count += 1
        print('DV finds counterexamples on {} small regions!'.format(count))
        # samples = np.array([x for x in samples if self.is_counter_example(B, x, 'diff')])
        # This is a bit time consuming, it can be masked if necessary.
        return samples, satisfy

    def generate_sample(self, center, sample_nums=10):
        result = []
        for i in range(sample_nums):
            rd = (np.random.random(self.n) - 0.5) * self.eps
            rd = rd + center
            result.append(rd)
        return result

    def is_counter_example(self, V, x, condition: str) -> bool:
        if condition not in ['V', 'DV']:
            raise ValueError(f'{condition} is not in validation condition!')

        d = {'x{}'.format(i + 1): x[i] for i in range(self.n)}
        v_numerical = V.subs(d)
        if condition == 'V' and v_numerical < 0:
            return True

        dot_v = sum([sp.diff(V, self.x[i]) * self.f[i](x) for i in range(self.n)])
        dot_v_numerical = dot_v.subs(d)

        if condition == 'DV' and dot_v_numerical > 0:
            return True
        return False

    def get_counter_example(self, V):

        S, satisfy1 = self.find_V(V, 100)
        samples = torch.Tensor(S)

        S, satisfy2 = self.find_diff(V, 10)
        samples = torch.cat([torch.Tensor(np.array(S)), samples], dim=0)

        return samples, (satisfy1 & satisfy2)

    def get_min_value(self, V, zones, V_x="", margin=0):
        m = gp.Model()
        x = m.addVars(range(1, self.n + 1), lb=[zones[i][0] for i in range(self.n)],
                      ub=[zones[i][1] for i in range(self.n)])

        V = re.sub(r"(" + 'x' + r")(\d*)", r"\1[\2]", V)
        m.setObjective(eval(V))

        if len(V_x) > 0:
            V_x = re.sub(r"(" + 'x' + r")(\d*)", r"\1[\2]", V_x)
            m.addConstr(eval(V_x) >= -margin)
            m.addConstr(eval(V_x) <= margin)

        # m.setParam('MIPFocus', 1)
        try:
            m.optimize()
        except gp.GurobiError:
            m.setParam('NonConvex', 2)
            m.optimize()

        status = False
        obj = 0
        x_values = []
        if m.status == gp.GRB.OPTIMAL:
            x_values = [i.X for i in m.getVars()]
            obj = m.getObjective().getValue()
            status = True
        return obj, x_values, status


if __name__ == '__main__':
    """

    test code!!

    """
    ex = get_example_by_id(2)
    # B = "6.3961523198475643 + 1.4535323575149564 * x1 + 1.4538974184013993 * x2 + 1.8135227930165725 * x3 +
    # 2.6902504633476734 * x4 + 3.2547335178420469 * x5 + 1.5728182938383033 * x6 + 1.4887045450483571 * x7 -
    # 0.098361374314026334 * (x1 * x2) + 0.61292489189007737 * (x1 * x3) + 0.22702188366685938 * (x1 * x4) -
    # 0.29694941092202465 * (x1 * x5) + 0.34135880540045682 * (x1 * x6) - 0.19533839339817044 * (x1 * x7) +
    # 0.14353724345397728 * (x2 * x3) - 0.52722748266477482 * (x2 * x4) - 0.19146523912842889 * (x2 * x5) -
    # 0.0014141270967824929 * (x2 * x6) + 0.1632478170467323 * (x2 * x7) - 1.7855562466953465 * (x3 * x4) +
    # 0.25579697171064852 * (x3 * x5) - 2.0831653388678513 * (x3 * x6) + 0.30607641216363024 * (x3 * x7) +
    # 1.2308796476983777 * (x4 * x5) + 0.058041930837321551 * (x4 * x6) - 0.23369116301049397 * (x4 * x7) -
    # 2.0995297184794013 * (x5 * x6) + 0.41654730845994981 * (x5 * x7) + 0.41695961994851316 * (x6 * x7) +
    # 0.59556693473144229 * x1^2 + 0.6707467357429745 * x2^2 + 1.2365762919341943 * x3^2 + 0.9766598185122628 * x4^2
    # + 0.99728844893602731 * x5^2 + 1.5976945732997536 * x6^2 + 0.43612154532794728 * x7^2"
    B = "28.0704151651177 * x1 ** 4 - 20.0106483286208 * x1 ** 3 * x2 + 26.465752726177 * x1 ** 3 - 10.5896550179601 " \
        "* x1 ** 2 * x2 ** 2 - 163.194332307303 * x1 ** 2 * x2 + 1.75982107627909 * x1 ** 2 - 55.7954417652044 * x1 * " \
        "x2 ** 3 - 117.098867814283 * x1 * x2 ** 2 - 232.300354924433 * x1 * x2 - 106.966918671716 * x1 - " \
        "19.76620098906 * x2 ** 4 - 15.1406211339187 * x2 ** 3 - 30.165728494359 * x2 ** 2 - 8.47711071469021 * x2 + " \
        "52.2056140217354"
    cef = CounterExampleFinder(ex)
    print(cef.get_counter_example(B))
