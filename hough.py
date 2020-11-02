import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
import os
import shutil
from PIL import Image
import glob


def vote(point, param_func, param_space, xvals, yvals):
    for u in range(len(xvals) - 1):
        for v in range(len(yvals) - 1):
            param_val_lowx = param_func(point[0], point[1], xvals[u])
            param_val_highx = param_func(point[0], point[1], xvals[u + 1])

            if (param_val_lowx >= yvals[v] and param_val_lowx <= yvals[v + 1]) or \
               (param_val_highx >= yvals[v] and param_val_highx <= yvals[v + 1]):
                param_space[v][u] += 1


def transform(points, param_func, param_dims, xmin, xmax, ymin, ymax):
    param_space = np.zeros(param_dims)
    yvals = np.arange(ymin, ymax + ((ymax - ymin) / param_dims[0]), (ymax - ymin) / param_dims[0])
    xvals = np.arange(xmin, xmax + ((xmax - xmin) / param_dims[1]), (xmax - xmin) / param_dims[1])
    print(xvals)
    print(yvals)
    for p in points:
        vote(p, param_func, param_space, xvals, yvals)
    return param_space


def plot_points(points, filename, title):
    plt.scatter(points[:,0], points[:,1], cmap='Greys')
    plt.title(title, loc='left')
    plt.savefig(filename)
    plt.clf()


def plot_param_space(param_space, filename, title):
    plt.imshow(param_space, cmap='gray')
    plt.gca().invert_yaxis()
    plt.title(title, loc='left')
    plt.savefig(filename)
    plt.clf()


def cartesian_param(xi, yi, m):
    return m * (-xi) + yi

def polar_param(xi, yi, theta):
    return xi * math.cos(theta) + yi * math.sin(theta)


points = np.array([(1, 0), (-1, 0), (0, 3)])

# compute polar parameter space
param_space = transform(points, polar_param, (60, 60), -3, 3, -6, 6)
print(np.flip(param_space, axis=0))
plot_param_space(param_space, 'polar_param_space', 'Polar Parameter Space')

# compute cartesian parameter space
param_space = transform(points, cartesian_param, (60, 60), -3, 3, -6, 6)
print(np.flip(param_space, axis=0))
plot_param_space(param_space, 'cartesian_param_space', 'Cartesian Parameter Space')

plot_points(points, 'points', 'Point Space')
