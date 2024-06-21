import numpy as np


class unconstrained_min:
    """
    At each iteration, the algorithm reports (prints to console) the iteration number 𝑖, the current location 𝑥𝑖,
    and the current objective value 𝑓(𝑥𝑖). by wanted method.
    :param f:function minimized
    :param x0: starting point
    :param obj_tol: numeric tolerance for successful termination in terms of small enough change in objective function
     values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)), or in the Newton Decrement based approximation of
     the objective decrease.
    :param param_tol:numeric tolerance for successful termination in terms of small enough distance between two
     consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖)
    :param max_iter: maximum allowed number of iterations.
    :return: final location, final objective value and a success/failure Boolean flag.
    """

    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.x_list = []
        self.f_x_list = []


    def minimaze(self, func, type):
        # implement newton algorithm on the function
        f, g, h = func(self.x0, True if type == "Newton" else 0)
        cur_x = self.x0
        self.x_list.append(self.x0)
        self.f_x_list.append(f)
        while len(self.x_list) < self.max_iter + 1:
            if type == "Newton":
                direction = -np.linalg.solve(h, g)
            elif type == "gradient":
                direction = -g
            print(f"Iteration {len(self.x_list) - 1}: f({cur_x}) = {f}")
            alpha = self.wolfe(func, cur_x, direction)
            cur_x = cur_x + alpha * direction
            f, g, h = func(cur_x,  True if type == "Newton" else 0)
            self.x_list.append(cur_x)
            self.f_x_list.append(f)

            if len(self.x_list)> 2 :
                if type == "Newton":
                    if np.linalg.norm(self.x_list[-2] - self.x_list[-1]) <= self.param_tol or (0.5 * direction.T @ (h @ direction)) ** 0.5< self.obj_tol:
                        break
                else:
                    if abs(self.f_x_list[-2] - self.f_x_list[-1]) <= self.obj_tol or np.linalg.norm(self.x_list[-2] - self.x_list[-1]) <= self.param_tol:
                        break


        succ = True if len(self.x_list) < self.max_iter else False
        if succ:
            print(f"{type} stoped at iteration {len(self.x_list)} successfully")
        else:
            print(f"{type} stoped at iteration {len(self.x_list)} Not successfully")
        return cur_x, f, succ

    def wolfe(self, func, x, direction):
        # calculation of wolf
        Wolfe_condition_constant = 0.01
        backtracking_constant = 0.5
        alpha = 1.0
        while func(x + alpha * direction, False)[0] > func(x, False)[0] + Wolfe_condition_constant * alpha * np.dot(func(x, False)[1], direction):
            alpha = alpha * backtracking_constant
        return alpha
