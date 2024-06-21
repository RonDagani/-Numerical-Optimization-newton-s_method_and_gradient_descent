import numpy as np
import matplotlib.pyplot as plt

def plot_graph(f, point_gd, point_newton, x_limit, y_limit, title):
    """
    A utility to create a plot, that given an objective function and limits for the 2D axes, plots the
    contour lines of the function.

    :param f: the relevant function
    :param point_gd: (x,y) coordinates of the gradient decent
    :param point_newton: (x,y) coordinates of newton
    :param x_limit: x limits of the graph
    :param y_limit: y limits of the graph
    :param title: name of the function
    """
    x_grad_dec = np.array(point_gd).squeeze()[:,0]
    y_grad_dec = np.array(point_gd).squeeze()[:,1]

    X = np.linspace(x_limit[0], x_limit[1], 100)
    Y = np.linspace(y_limit[0], y_limit[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]), False)[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    fig = plt.figure()
    if title=='linear':
        level = 100
    elif title=='rosenbrock':
        level = 750
    else:
        level = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 35)
    ct = plt.contour(X, Y, Z, levels=level)

    plt.colorbar(ct)

    plt.plot(x_grad_dec, y_grad_dec,'-p', label="gradient descent")
    if point_newton != None:
        x_newton = np.array(point_newton).squeeze()[:, 0]
        y_newton = np.array(point_newton).squeeze()[:, 1]
        plt.plot(x_newton, y_newton, '-s', label="newton")
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.legend()
    plt.title(label=title)
    plt.show()
    fig.savefig(f'{title}.png')



def plot_graph_iterations(fx_gd, fx_newton, title=None):
    """
     A utility that plots function values at each iteration, for given methods (on the same, single
    plots) to enable comparison of the decrease in function values of methods.
    :param fx_gd: f(x) gradient decent
    :param fx_newton: f(x) newton
    :param title: name of function
    """
    fig = plt.figure()
    plt.plot(np.array(fx_gd).squeeze(), label="gradient descent")
    if fx_newton != None:
        plt.plot(np.array(fx_newton).squeeze(), label="newton")
    plt.xlabel("iteration")
    plt.ylabel("fx value")
    plt.title(label=title)
    plt.legend()
    plt.show()
    fig.savefig(f'{title}_xy.png')


