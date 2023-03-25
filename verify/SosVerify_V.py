import sympy as sp
from SumOfSquares import SOSProblem
from benchmarks.Exampler_V import Example, get_example_by_id
from functools import reduce
from itertools import product
import time


class SosValidator_V():
    def __init__(self, example: Example, V) -> None:
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(example.n)])
        self.n = example.n
        self.Invs = example.D_zones
        self.f = [example.f[i](self.x) for i in range(self.n)]
        self.V = V
        self.var_count = 0

    def polynomial(self, deg=2):  # Generating polynomials of degree n-ary deg.
        parameters = []
        terms = []
        exponents = list(product(range(deg + 1), repeat=self.n))  # Generate all possible combinations of indices.
        exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
        poly = 0
        for e in exponents:  # Generate all items.
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            terms.append(reduce(lambda a, b: a * b, [self.x[i] ** exp for i, exp in enumerate(e)]))
            poly += parameters[-1] * terms[-1]
        return poly, parameters, terms

    def SovleV(self, deg=2):
        prob_unsafe = SOSProblem()
        V = self.V
        x = self.x
        Invs = self.Invs
        for i in range(self.n):
            Qi, parameters, terms = self.polynomial(deg)
            prob_unsafe.add_sos_constraint(Qi, x)
            V = V + Qi * (x[i] - Invs[i][0]) * (x[i] - Invs[i][1])
        V = sp.expand(V)
        prob_unsafe.add_sos_constraint(V, x)
        try:
            prob_unsafe.solve(solver='mosek')
            return True
        except:
            return False

    def SolveDiffV(self, deg=2):
        prob_inv = SOSProblem()
        x = self.x
        Invs = self.Invs
        V = self.V
        DV = -sum([sp.diff(V, x[i]) * self.f[i] for i in range(self.n)])
        for i in range(self.n):
            Si, parameters, terms = self.polynomial(deg)
            prob_inv.add_sos_constraint(Si, x)
            DV = DV + Si * (x[i] - Invs[i][0]) * (x[i] - Invs[i][1])

        DV = sp.expand(DV)
        prob_inv.add_sos_constraint(DV, x)
        try:
            prob_inv.solve(solver='mosek')
            return True
        except:
            return False

    def SolveAll(self, deg=(2, 2)):
        assert len(deg) == 2

        isV = self.SovleV(deg[0])
        if not isV:
            print('V is not satisfied.')

        isDV = self.SolveDiffV(deg[1])
        if not isDV:
            print('DV is not satisfied.')

        return isV & isDV


if __name__ == '__main__':
    """
    test code!!
    """

    # ex1
    B = "(6.3961523198475643 + 1.4535323575149564 * x1 + 1.4538974184013993 * x2 + 1.8135227930165725 * x3 + " \
        "2.6902504633476734 * x4 + 3.2547335178420469 * x5 + 1.5728182938383033 * x6 + 1.4887045450483571 * x7 - " \
        "0.098361374314026334 * (x1 * x2) + 0.61292489189007737 * (x1 * x3) + 0.22702188366685938 * (x1 * x4) - " \
        "0.29694941092202465 * (x1 * x5) + 0.34135880540045682 * (x1 * x6) - 0.19533839339817044 * (x1 * x7) + " \
        "0.14353724345397728 * (x2 * x3) - 0.52722748266477482 * (x2 * x4) - 0.19146523912842889 * (x2 * x5) - " \
        "0.0014141270967824929 * (x2 * x6) + 0.1632478170467323 * (x2 * x7) - 1.7855562466953465 * (x3 * x4) + " \
        "0.25579697171064852 * (x3 * x5) - 2.0831653388678513 * (x3 * x6) + 0.30607641216363024 * (x3 * x7) + " \
        "1.2308796476983777 * (x4 * x5) + 0.058041930837321551 * (x4 * x6) - 0.23369116301049397 * (x4 * x7) - " \
        "2.0995297184794013 * (x5 * x6) + 0.41654730845994981 * (x5 * x7) + 0.41695961994851316 * (x6 * x7) + " \
        "0.59556693473144229 * x1**2 + 0.6707467357429745 * x2**2 + 1.2365762919341943 * x3**2 + 0.9766598185122628 * " \
        "x4**2 + 0.99728844893602731 * x5**2 + 1.5976945732997536 * x6**2 + 0.43612154532794728 * x7**2)"

    # Validator = SosValidator_B(get_example_by_id(1), B=sp.simplify(B))
    # t1 = time.time()
    # print('Init validation results:', Validator.SovleInit())
    # print('Unsafe validation results:', Validator.SovleUnsafe())
    # print('Lie derivative validation results:', Validator.SolveDiffB())
    # t2 = time.time()
    # print('validation time:{} s'.format(round(t2 - t1, 2)))
