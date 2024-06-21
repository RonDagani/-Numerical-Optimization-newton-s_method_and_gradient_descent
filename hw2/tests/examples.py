import numpy as np
import math


def circles(x, flag_hessian):
    # example 1
    q = np.array([[1, 0], [0, 1]])
    f = x.T @ q @ x
    g = 2 * q @ x

    if flag_hessian:
        return f, g, 2*q
    return f, g, None


def ellipses(x, flag_hessian):
    # example 2
    q = np.array([[1, 0], [0, 100]])
    f = x.T @ q @ x
    g = 2 * q @ x

    if flag_hessian:
        return f, g, 2 * q
    return f, g, None


def rotated_ellipses(x, flag_hessian):
    # example 3
    q = np.dot(np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]]).T,
           np.dot(np.array([[100, 0], [0, 1]]),
                  np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])))
    f = x.T @ q @ x
    g = 2 * q @ x

    if flag_hessian:
        return f, g, 2 * q
    return f, g, None


def rosenbrock(x, flag_hessian):
    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    if flag_hessian:
        h=np.array([[-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]]).squeeze()
        return f, g, h
    return f, g, None

def linear(x, flag_hessian):
    a = np.array([1,2])
    f = np.dot(a.T, x)
    g = a

    if flag_hessian:
        return f, g, 0
    return f, g, None


def exp_function(x, flag_hessian):
    f = np.exp(x[0]+3*x[1]-0.1)+np.exp(x[0]-3*x[1]-0.1)+np.exp(-x[0]-0.1)
    g = np.array([np.exp(x[0]+3*x[1]-0.1)+np.exp(x[0]-3*x[1]-0.1)-np.exp(-x[0]-0.1),
                  3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1)])
    if flag_hessian:
        a1 = np.exp(x[0]+3*x[1]-0.1)+np.exp(x[0]-3*x[1]-0.1)+np.exp(-x[0]-0.1)
        a2 = 3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1)
        a3 = 3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1)
        a4 = 9*np.exp(x[0]+3*x[1]-0.1)+9*np.exp(x[0]-3*x[1]-0.1)
        h = np.array([[a1, a2], [a3, a4]]).squeeze()

        return f, g, h
    return f, g, None


if __name__ == '__main__':
    main()
