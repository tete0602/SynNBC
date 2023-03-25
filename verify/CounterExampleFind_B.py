import re
import gurobipy as gp
import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize, NonlinearConstraint
from benchmarks.Exampler_B import Example, get_example_by_name
from utils.Config_B import CegisConfig


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
        if n > 5 and np.random.random() > 0.5:
            subbounds = split_bounds(bounds, n + 1)
        else:
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
    def __init__(self, example: Example, config: CegisConfig):
        self.n = example.n
        self.I_zones = example.I_zones
        self.U_zones = example.U_zones
        self.D_zones = example.D_zones
        self.f = example.f
        self.choice = config.CHOICE
        self.eps = 0.05
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.split_D_zones = split_bounds(np.array(self.D_zones), 0)
        self.config = config

    def find_init(self, B, sample_nums=10):
        samples = []
        satisfy = True

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[0] == 1:
            B_str = str(B)
            B_str = '-(' + B_str + ')'
            res, x_values, status = self.get_min_value(B_str, self.I_zones)
            x_values = np.array(x_values)
            if -res < 0 or status == False:
                pass
            else:
                print('Counterexamples are found in the initial set:', x_values, 'B:', B)
                samples.append(x_values)
                samples.extend(self.generate_sample(x_values, sample_nums - 1))
                satisfy = False
        elif self.choice[0] == 0:
            opt = sp.lambdify(self.x, B)
            # Set an initial guess.
            x0 = np.zeros(shape=self.n)
            res = minimize(fun=lambda x: -opt(*x), x0=x0, bounds=self.I_zones)
            if -res.fun < 0:
                pass
            else:
                print('Counterexamples found in the initial set:', res.x, 'B:', B)
                samples.append(res.x)
                samples.extend(self.generate_sample(res.x, sample_nums - 1))
                satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(B, x, 'init')])
        return samples, satisfy

    def find_unsafe(self, B, sample_nums=10):
        samples = []
        satisfy = True

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[1] == 1:
            B_str = str(B)
            res, x_values, status = self.get_min_value(B_str, self.U_zones)
            x_values = np.array(x_values)
            if res > 0 or status == False:
                pass
            else:
                print('Counterexamples found in the unsafe set:', x_values)
                samples.append(x_values)
                samples.extend(self.generate_sample(x_values, sample_nums - 1))
                satisfy = False
        elif self.choice[1] == 0:
            opt = sp.lambdify(self.x, B)
            x0 = np.zeros(shape=(self.n))
            res = minimize(fun=lambda x: opt(*x), x0=x0, bounds=self.U_zones)

            if res.fun > 0:
                pass
            else:
                print('Counterexamples found in the unsafe set:', res.x)
                samples.append(res.x)
                samples.extend(self.generate_sample(res.x, sample_nums - 1))
                satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(B, x, 'unsafe')])
        return samples, satisfy

    def find_diff(self, B, sample_nums=10):
        samples = []
        satisfy = True
        count = 0

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[2] == 1:
            B_str = str(B)
            x = self.x
            DB = sum([sp.diff(B, x[i]) * self.f[i](x) for i in range(self.n)])
            DB = sp.expand(DB)
            DB_str = str(DB)
            DB_str = '-(' + DB_str + ')'
            # The condition of B(x)==0 is relaxed to find counterexamples,
            # and the existence of counterexamples does not mean that sos must not be satisfied.
            margin = 0.00
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones

            for bound in bounds:
                res, x_values, status = self.get_min_value(DB_str, bound, B_x=B_str, margin=margin)
                x_values = np.array(x_values)

                if -res < 0 or status == False:
                    pass
                else:
                    samples.append(x_values)
                    samples.extend(self.generate_sample(x_values, sample_nums - 1))
                    satisfy = False
                    count += 1
        elif self.choice[2] == 0:
            opt = sp.lambdify(self.x, B)
            x = self.x
            DB = sum([sp.diff(B, x[i]) * self.f[i](x) for i in range(self.n)])
            optDB = sp.lambdify(x, DB)
            # The condition of B(x)==0 is relaxed to find counterexamples,
            # and the existence of counterexamples does not mean that sos must not be satisfied.
            margin = 0.00
            constraint = NonlinearConstraint(lambda x: opt(*x), -margin, margin)
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones

            for bound in bounds:
                x0 = (bound.T[0] + bound.T[1]) / 2
                res = minimize(fun=lambda x: -optDB(*x), x0=x0, bounds=bound, constraints=constraint)
                if -res.fun < 0:
                    pass
                else:
                    samples.append(res.x)
                    samples.extend(self.generate_sample(res.x, sample_nums - 1))
                    satisfy = False
                    count += 1
        print('Lie derivative finds counterexamples on {} small regions!'.format(count))
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

    def is_counter_example(self, B, x, condition: str) -> bool:
        if condition not in ['init', 'unsafe', 'diff']:
            raise ValueError(f'{condition} is not in validation condition!')
        d = {'x{}'.format(i + 1): x[i] for i in range(self.n)}
        b_numerical = B.subs(d)
        dot_b = sum([sp.diff(B, self.x[i]) * self.f[i](x) for i in range(self.n)])
        dot_b_numerical = dot_b.subs(d)
        if condition == 'init' and b_numerical > 0:
            return True
        if condition == 'unsafe' and b_numerical < 0:
            return True
        if condition == 'diff' and dot_b_numerical > 0:
            return True
        return False

    def get_counter_example(self, B):
        samples = []
        S, satisfy1 = self.find_init(B, 100)
        samples.append(torch.Tensor(S))
        S, satisfy2 = self.find_unsafe(B, 100)
        samples.append(torch.Tensor(S))
        S, satisfy3 = self.find_diff(B, 20 if self.config.SPLIT_D else 100)
        samples.append(torch.Tensor(np.array(S)))
        # print(satisfy1,satisfy2,satisfy3)
        return samples, (satisfy1 & satisfy2 & satisfy3)

    def get_min_value(self, B, zones, B_x="", margin=0):
        m = gp.Model()
        x = m.addVars(range(1, self.n + 1), lb=[zones[i][0] for i in range(self.n)],
                      ub=[zones[i][1] for i in range(self.n)])

        B = re.sub(r"(" + 'x' + r")(\d*)", r"\1[\2]", B)
        m.setObjective(eval(B))

        if len(B_x) > 0:
            B_x = re.sub(r"(" + 'x' + r")(\d*)", r"\1[\2]", B_x)
            m.addConstr(eval(B_x) >= -margin)
            m.addConstr(eval(B_x) <= margin)

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
    ex = get_example_by_name('emsoft_c11_4_2')
    B = "-1.98214774297e-10-1.25924660981e-12*x2-9.74452688236e-12*x5-9.74357368155e-12*x7+1.1449887559e-13*x10-6" \
        ".8936332256e-11*x2^2-1.46311839551e-09*x5^2-1.46311380156e-09*x7**2-2.15799504357e-09*x10**2-3.19654158345e-10" \
        "*x2*x5-3.19629800246e-10*x2*x7-2.56048044107e-09*x5*x7-2.00084051036e-12*x2*x10-1.07416966319e-10*x5*x10-1" \
        ".0742096333e-10*x7*x10 "
    B = B.replace('x2', 'x1').replace('x5', 'x2').replace('x7', 'x3').replace('x10', 'x4')

    CegisConfig.SPLIT_D = True
    cef = CounterExampleFinder(ex, CegisConfig)
    cef.get_counter_example(B)
