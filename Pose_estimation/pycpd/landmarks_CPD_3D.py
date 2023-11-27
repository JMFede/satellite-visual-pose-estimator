import argparse
from functools import partial
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from pycpd import RigidRegistration
from scipy.spatial.transform import Rotation
import numpy as np
import os
import math

def visualize(iteration, error, X, Y, ax, fig, save_fig=False):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.view_init(90, -90)
    if save_fig is True:
        ax.set_axis_off()

    plt.draw()
    if save_fig is True:
        os.makedirs("../Coherent_Point_Drift/images/landmarks/", exist_ok=True)
        fig.savefig("../Coherent_Point_Drift/images/landmarks/landmark0_{:04}.tiff".format(iteration), dpi=600)  # Used for making gif.
        fig.savefig("../Coherent_Point_Drift/images/landmarks/landmark0_{:04}.png".format(iteration), dpi=600)
    plt.pause(0.1)


def CPD(show=True, save=False, prediction=None, visibility=None, visibility_path='../visibility.txt', prediction_path='../prediction.txt', ground_truth_path='../rigid_shape.txt'):
    #print(save)
    if prediction is None or visibility is None:
        prediction = np.loadtxt(prediction_path)
        visibility = np.loadtxt(visibility_path)

    idx = np.where(visibility==1)[0]
    X = prediction
    X = X[idx,:]

    Y = np.loadtxt(ground_truth_path)
    Y = Y[idx,:]

    reg = RigidRegistration(**{'X': X, 'Y': Y})

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        callback = partial(visualize, ax=ax, fig=fig, save_fig=save if type(save) is list else save)
        reg.register(callback)
        euler_angles = Rotation.from_matrix(reg.R).as_euler('xyz', degrees=True)
        print('Translation: ', reg.t)
        print("Euler Angles (Roll, Pitch, Yaw) in degree:", euler_angles)
        plt.show()
    else:
        reg.register()
        euler_angles = Rotation.from_matrix(reg.R).as_euler('xyz', degrees=True)
    
    return euler_angles, reg.t


if __name__ == '__main__':
    CPD(show=True, save=True)

