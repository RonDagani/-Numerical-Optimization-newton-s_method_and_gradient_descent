import unittest
import tests.examples
from src.unconstrained_min import *
from src.utils import *
import numpy as np

class TestStringMethods(unittest.TestCase):

    x0 = np.array([1, 1])
    step_tolerance = 10 ** -12
    numeric_tolerances = 10 ** -8
    max_iterations = 100


    def test_circles(self):
        name = 'Quadratic_circles'
        cur_fun = tests.examples.circles
        test_fun(name, cur_fun, self.x0, self.step_tolerance, self.numeric_tolerances, self.max_iterations, limit = 3)

    def test_ellipses(self):
        name = 'Quadratic_ellipses'
        cur_fun = tests.examples.ellipses
        test_fun(name, cur_fun, self.x0, self.step_tolerance, self.numeric_tolerances, self.max_iterations, limit = 3)


    def test_rotated_ellipses(self):
        name = 'Quadratic_rotated_ellipses'
        cur_fun = tests.examples.rotated_ellipses
        test_fun(name, cur_fun, self.x0, self.step_tolerance, self.numeric_tolerances, self.max_iterations, limit = 1.5)


    def test_exp_function(self):
        name = 'exp_function'
        cur_fun = tests.examples.exp_function
        test_fun(name, cur_fun, self.x0, self.step_tolerance, self.numeric_tolerances, self.max_iterations, limit = 1.5)

    def test_linear(self):
        name = 'linear'
        cur_fun = tests.examples.linear
        test_fun(name, cur_fun, self.x0, self.step_tolerance, self.numeric_tolerances, self.max_iterations, limit = 250)

    def test_rosenbrock(self):
        x0 = np.array([-1, 2])
        name = 'rosenbrock'
        cur_fun = tests.examples.rosenbrock
        max_iter = {'gradient':10000, 'newton':100}
        test_fun(name, cur_fun, x0, self.step_tolerance, self.numeric_tolerances, max_iter, limit = 2.5)


def test_fun(name, cur_fun, x0, step_tolerance, numeric_tolerances, max_iterations, limit):
    print('Testing ' + name)
    minimizer = {}
    if isinstance(max_iterations, int):
        max_iterations_dict = {'newton':max_iterations, 'gradient':max_iterations}
    else:
        max_iterations_dict = max_iterations

    minimizer["gradient"] = unconstrained_min(cur_fun, x0, step_tolerance, numeric_tolerances,
                                              max_iterations_dict["gradient"])
    _, _, _ = minimizer["gradient"].minimaze(cur_fun, "gradient")

    if name != 'linear':
        minimizer["newton"] = unconstrained_min(cur_fun, x0, step_tolerance, numeric_tolerances,
                                            max_iterations_dict["newton"])
        _, _, _ = minimizer["newton"].minimaze(cur_fun, "Newton")
        plot_graph(cur_fun, point_gd=minimizer['gradient'].x_list, point_newton=minimizer['newton'].x_list,
                   x_limit=[-limit, limit], y_limit=[-limit, limit], title=name)
        plot_graph_iterations(minimizer['gradient'].f_x_list, minimizer['newton'].f_x_list, name)

    else:
        plot_graph(cur_fun, point_gd=minimizer['gradient'].x_list, point_newton=None,
                   x_limit=[-limit, limit], y_limit=[-limit, limit], title=name)
        plot_graph_iterations(fx_gd=minimizer['gradient'].f_x_list, title=name, fx_newton=None)
    print('\n')


if __name__ == '__main__':
    unittest.main()
