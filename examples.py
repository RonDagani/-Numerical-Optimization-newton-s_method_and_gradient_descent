import numpy as np


def qp(x, hassian = False):
    # objective function
    f = x[0] ** 2  + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])
    if hassian:
        h = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        return f, g, h
    return f, g, None

def qp_ineq_1(x, hassian = False):
    f = -x[0]
    g = np.array([-1, 0, 0])
    if hassian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h
    return f, g, None


def qp_ineq_2(x, hassian = False):
    f = -x[1]
    g = np.array([0, -1, 0])
    if hassian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h
    return f,g, None

def qp_ineq_3(x, hassian = False):
    f = -x[2]
    g = np.array([0, 0, -1])
    if hassian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h
    return f,g, None


def lp(x, hassian = False):
    # objective function
    f = -x[0] - x[1]
    g = np.array([-1, -1])
    if hassian:
        h = np.array([
            [0,0],
            [0,0]
        ])
        return f, g, h
    return f,g, None


def lp_ineq1(x, hassian = False):
    f = x[1] -1
    g = np.array([0, 1])
    if hassian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
        return f, g, h
    return f,g, None

def lp_ineq2(x, hassian = False):
    f = x[0] -2
    g = np.array([1, 0])
    if hassian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
        return f, g, h
    return f,g, None

def lp_ineq_3(x, hassian = False):
    f = -x[1]
    g = np.array([0, -1])
    if hassian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
        return f, g, h
    return f, g, None

def lp_ineq_4(x, hassian = False):
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1])
    if hassian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
        return f, g, h
    return f,g, None