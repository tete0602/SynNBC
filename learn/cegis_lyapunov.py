import torch
import numpy as np
from utils.Config_V import CegisConfig
from learn.net_V import Learner
from verify.CounterExampleFind_V import CounterExampleFinder
from verify.SosVerify_V import SosValidator_V
import timeit


class Cegis:

    def __init__(self, config: CegisConfig):
        n = config.EXAMPLE.n
        self.ex = config.EXAMPLE
        self.n = n
        self.f = config.EXAMPLE.f
        self.D_zones = np.array(config.EXAMPLE.D_zones, dtype=np.float32).T
        self.batch_size = config.BATCH_SIZE
        self.learning_rate = config.LEARNING_RATE
        self.Learner = Learner(config)

        self.optimizer = config.OPT(self.Learner.net.parameters(), lr=self.learning_rate)
        self.CounterExampleFinder = CounterExampleFinder(config.EXAMPLE, config)

        self.max_cegis_iter = 100
        self.DEG = config.DEG

    def solve(self):

        S, Sdot = self.generate_data()
        # the CEGIS loop
        deg = self.DEG
        t_learn = 0
        t_cex = 0
        t_sos = 0
        for i in range(self.max_cegis_iter):
            t1 = timeit.default_timer()
            self.Learner.learn(self.optimizer, S, Sdot)
            t2 = timeit.default_timer()
            t_learn += t2 - t1
            V = self.Learner.net.get_lyapunov()

            print(f'iter: {i + 1} \nV = {V}')

            t3 = timeit.default_timer()
            Sos_Validator = SosValidator_V(self.ex, V)
            t4 = timeit.default_timer()
            t_sos += t4 - t3
            if Sos_Validator.SolveAll(deg=deg):
                print('SOS verification passed!')
                break

            # In the negative example of Lie derivative, the condition of B(x)==0 is relaxed to |B(x)|<=margin
            # to find a negative example, so the existence of a negative example does not mean that sos must
            # not be satisfied
            t5 = timeit.default_timer()
            samples, satisfy = self.CounterExampleFinder.get_counter_example(V)
            t6 = timeit.default_timer()
            t_cex += t6 - t5
            if satisfy:
                print('No counterexamples were found!')

            if satisfy:
                # If no counterexample is found, but SOS fails, it may be that the number of multipliers is too low.
                deg[1] += 2

            S, Sdot = self.add_ces_to_data(S, Sdot, samples)
            print('-' * 200)
        print('Total learning time:{}'.format(t_learn))
        print('Total counter-examples generating time:{}'.format(t_cex))
        print('Total sos verifying time:{}'.format(t_sos))

    def add_ces_to_data(self, S, Sdot, ces):

        print(f'Add {len(ces)} counterexamples!')
        S = torch.cat([S, ces], dim=0).detach()
        dot_ces = self.x2dotx(ces)
        Sdot = torch.cat([Sdot, dot_ces], dim=0).detach()

        return S, Sdot

    def generate_data(self):

        D_len = self.D_zones[1] - self.D_zones[0]

        # Let more sample points fall on the boundary.
        size = 2
        S = torch.clip((torch.rand([self.batch_size, self.n]) - 0.5) * size, -0.5, 0.5) + 0.5
        S = S * D_len + self.D_zones[0]

        Sdot = self.x2dotx(S)

        return S, Sdot

    def x2dotx(self, X):  # todo Changing to map mapping will be faster
        f_x = []
        for x in X:
            f_x.append([self.f[i](x) for i in range(self.n)])
        return torch.Tensor(f_x)


if __name__ == '__main__':
    pass
