import unittest
import numpy as np
from tests.examples import *
from src.constrained_min import interior_pt
from src.utils import plot_qp, plot_lp, iterations_graph


def print_results(fc, fo, f_ineq, path, qp):
    if qp:
        print("qp")

    else:
        print("lp:")
    print('Final candidate:', fc)
    print('Objective function of final candidate:', fo)
    print('Inequality constraints of final candidate:', f_ineq)
    final_path = np.array(path['points'])

    if qp:
        A = np.array([1, 1, 1]).reshape(1, 3)
        print('Equality constraints of final candidate:', (A * fc).sum())
        iterations_graph(path['fx_outter'], path['fx_inner'], 'Objective as a function of iteration - qp')
        plot_qp(final_path, 'Feasible region and algorithm path - qp ')

    else:
        iterations_graph(path['fx_outter'], path['fx_inner'], 'Objective as a function of iteration - lp')
        plot_lp(final_path, 'Feasible region and algorithm path - lp ')


class TestConstrained(unittest.TestCase):
    def test_qp(self):
        A = np.array([1, 1, 1]).reshape(1, 3)
        x0 = np.array([0.1, 0.2, 0.7])
        ineq_constraints_qp = [qp_ineq_1, qp_ineq_2, qp_ineq_3]

        fc, fo, path = interior_pt(qp, ineq_constraints_qp, A, 0, x0)
        f_ineq = [c(fc)[0] for c in ineq_constraints_qp]

        print_results(fc, fo, f_ineq, path, qp=True)

    def test_lp(self):
        x0 = np.array([0.5, 0.75])
        ineq_constraints_lp = [lp_ineq1, lp_ineq2, lp_ineq_3, lp_ineq_4]

        fc, fo, path = interior_pt(lp, ineq_constraints_lp, None, 0, x0)
        f_ineq = [c(fc)[0] for c in ineq_constraints_lp]

        print_results(fc, fo, f_ineq, path, qp=False)

