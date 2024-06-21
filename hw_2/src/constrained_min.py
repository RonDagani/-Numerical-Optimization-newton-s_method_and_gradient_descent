import numpy as np
import math


def wolfe(f, x, f_val, grad_val, direction, ineq_constraints, tau, max_iter=15):
    iteration = 0
    alpha = 1
    Wolfe_condition_constant = 0.01
    backtracking_constant = 0.5

    while iteration < max_iter and f(x + alpha * direction)[0] > f_val + Wolfe_condition_constant * alpha * grad_val.dot(direction):
        alpha *= backtracking_constant
        iteration += 1

    return alpha


def log_bar(ineq_constraints, x0):
    x_dim = x0.shape[0]
    log_f = 0
    log_g = np.zeros((x_dim,))
    log_h = np.zeros((x_dim, x_dim))

    for constraint in ineq_constraints:
        f_val, g_val, h_val = constraint(x0, hassian=True)
        log_f += math.log(-f_val)
        log_g += (1.0 / -f_val) * g_val

        grad_val = g_val / f_val
        grad_dim = grad_val.shape[0]
        grad_tile = np.tile(grad_val.reshape(grad_dim, -1), (1, grad_dim)) * np.tile(grad_val.reshape(grad_dim, -1).T,
                                                                                     (grad_dim, 1))
        log_h += (h_val * f_val - grad_tile) / f_val ** 2

    return -log_f, log_g, -log_h


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    num_constraints = len(ineq_constraints)
    path_info = {'points': [], 'fx_outter': [], 'fx_inner':[]}
    current_x = x0
    t = 1
    mu = 10

    initial_val, initial_grad, initial_hessian = func(current_x, hassian=True)
    log_f, log_g, log_h = log_bar(ineq_constraints, current_x)
    prev_f, prev_grad, prev_hessian = t * initial_val + log_f, t * initial_grad + log_g, t * initial_hessian + log_h
    path_info['points'].append(current_x.copy())
    path_info['fx_outter'].append(func(current_x.copy())[0])
    path_info['fx_inner'].append(func(current_x.copy())[0])


    while (num_constraints / t) > 1e-8:
        for _ in range(15):
            if eq_constraints_mat is not None:
                lhs_matrix = np.block([[prev_hessian, eq_constraints_mat.T], [eq_constraints_mat, 0]])
                rhs_vector = np.block([[-prev_grad, 0]])
                rhs_vector_t = rhs_vector.T
                solution = np.linalg.solve(lhs_matrix, rhs_vector_t).T[0]
                direction = solution[:eq_constraints_mat.shape[1]]
            else:
                direction = np.linalg.solve(prev_hessian, -prev_grad)
            step_size = wolfe(func, current_x, prev_f, prev_grad, direction, ineq_constraints, t)
            next_x = current_x + direction * step_size

            next_val, next_grad, next_hessian = func(next_x, hassian=True)
            log_f, log_g, log_h = log_bar(ineq_constraints, next_x)
            next_f, next_grad, next_hessian = t * next_val + log_f, t * next_grad + log_g, t * next_hessian + log_h
            if 0.5 * (np.sqrt(np.dot(direction, np.dot(next_hessian, direction.T))) ** 2) < 1e-8:
                break
            path_info['fx_inner'].append(func(current_x.copy())[0])
            current_x = next_x
            prev_f = next_f
            prev_grad = next_grad
            prev_hessian = next_hessian

        path_info['points'].append(current_x.copy())
        path_info['fx_outter'].append(func(current_x.copy())[0])
        t *= mu

    return current_x, func(current_x.copy())[0], path_info


