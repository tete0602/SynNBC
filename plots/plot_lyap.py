import numpy as np
import matplotlib.pyplot as plt
from plots.plot_fcns import plotting_3d, vector_field
import sympy as sp
from benchmarks.Exampler_V import Example
import os


def set_title_and_label_3d(ax, x_label, y_label, z_label, title):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)


def plot_benchmark_2d(ex: Example, V):
    assert (ex.n == 2)
    path = 'img/'
    if not os.path.exists(path):
        os.mkdir(path)

    x = sp.symbols(['x{}'.format(i + 1) for i in range(ex.n)])
    Vdot = sum([sp.diff(V, x[i]) * ex.f[i](x) for i in range(ex.n)])
    plot_limit = ex.D_zones[0, 1]
    X = np.linspace(*ex.D_zones[0], 100)
    Y = np.linspace(*ex.D_zones[1], 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v(x0, x1)

    ax = plotting_3d(x0, x1, plot_v)
    set_title_and_label_3d(ax, '$x_1$', '$x_2$', 'V', 'Lyapunov function')

    plt.savefig(path + ex.name + '_lyapunov3d.pdf')
    lambda_vdot = sp.lambdify(x, str(Vdot), modules=['numpy'])
    plot_vdot = lambda_vdot(x0, x1)

    ax = plotting_3d(x0, x1, plot_vdot)
    set_title_and_label_3d(ax, '$x_1$', '$x_2$', '$\dot{V}$', 'Lyapunov derivative')
    ################################
    # PLOT 2D -- CONTOUR
    ################################
    plt.savefig(path+ex.name+'_lyapunov_derivative3d.pdf')
    plt.figure()
    ax = plt.gca()

    # plot vector field
    xv = np.linspace(*ex.D_zones[0], 10)
    yv = np.linspace(*ex.D_zones[1], 10)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(ex.f, Xv, Yv, t)

    ax.contour(X, Y, plot_v, 5, linewidths=2, colors='k')
    plt.title('Lyapunov Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig(path + ex.name + '_lyapunov2d.pdf')
    plt.show()


def plot_lyapunov_fcn(x, V, f):
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_f = sp.lambdify(x, str(V), modules=['numpy'])
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v([x0], [x1])

    ax = plotting_3d(x0, x1, plot_v)
    set_title_and_label_3d(ax, '$x$', '$y$', 'V', 'Lyapunov function')

    ################################
    # PLOT 2D -- CONTOUR
    ################################

    # plt.figure()
    # ax = plt.gca()

    # plot vector field
    # xv = np.linspace(-plot_limit, plot_limit, 10)
    # yv = np.linspace(-plot_limit, plot_limit, 10)
    # Xv, Yv = np.meshgrid(xv, yv)
    # t = np.linspace(0, 5, 100)
    # vector_field(f, Xv, Yv, t)
    # ax.contour(X, Y, plot_v, 5, linewidths=2, colors='k')
    # plt.title('Lyapunov Border')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.show()
