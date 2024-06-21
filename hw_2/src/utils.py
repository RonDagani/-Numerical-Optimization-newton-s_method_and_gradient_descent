import matplotlib.pyplot as plt
import numpy as np


def plot_qp(path, title):
    fc = path[-1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #     ax.view_init(45,45) ##############################
    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], alpha=0.3)
    ax.scatter(fc[0], fc[1], fc[2], marker='o', label='final candidate')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.legend()
    plt.show()


def plot_lp(path, title):
    fig, ax = plt.subplots(1, 1)

    x = np.linspace(-1, 3, 300)
    y = np.linspace(-2, 2, 300)
    contraints = {  'y=0': (x, x*0),
                    'y=1': (x, x*0 + 1),
                    'x=2': (y*0 + 2, y),
                    'y=-x+1': (x, -x + 1)}

    for contraint, point in contraints.items():
        ax.plot(point[0], point[1], label=contraint)

    ax.plot(path[:, 0], path[:, 1], c='pink', label='path')
    ax.scatter(path[-1][0], path[-1][1], s=60, c='b', marker='o', label='final candidate')

    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], label='feasible region')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()


def iterations_graph(values_ouuer, values_inner, title):
    plt.plot(values_ouuer, label='outer values')
    # plt.plot(values_inner, label='inner values')
    plt.xlabel('iterations')
    plt.ylabel('fx')
    plt.legend()
    plt.title(title)
    plt.show()


