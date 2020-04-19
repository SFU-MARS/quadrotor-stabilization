import numpy as np
import math
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator

import matlab.engine as MatEng
import time
import pickle


class ttr_helper(object):
    def __init__(self, task='TakeOffHover'):
        self.task = task
        self.Vx = (-5, 5)
        self.Vy = (-5, 5)
        self.Vz = (-5, 5)
        self.Theta = (-np.pi/2, np.pi/2)
        self.Wt = (-10 * np.pi, 10 * np.pi)
        self.Phi = (-np.pi/2, np.pi/2)
        self.Wp = (-10 * np.pi, 10 * np.pi)
        self.ranges = np.array([self.Vx, self.Vy, self.Vz, self.Theta, self.Wt, self.Phi, self.Wp])
        self.state_step_num = np.array([101, 101, 101, 18, 201, 18, 201])

        self.ttrVxVzThetaWt_filepath = "/local-scratch/xlv/quad_stabilization/ttr_engine/{}/ttrVxVzThetaWt/ttrVxVzThetaWt.mat".format(self.task)
        self.ttrVyVzPhiWp_filepath   = "/local-scratch/xlv/quad_stabilization/ttr_engine/{}/ttrVyVzPhiWp/ttrVyVzPhiWp.mat".format(self.task)
        print(self.ttrVxVzThetaWt_filepath)

        self.ttrVxVzThetaWt_interp_filepath = "/local-scratch/xlv/quad_stabilization/ttr_engine/{}/ttrVxVzThetaWt/ttrVxVzThetaWt_interp.pkl".format(self.task)
        self.ttrVyVzPhiWp_interp_filepath   = "/local-scratch/xlv/quad_stabilization/ttr_engine/{}/ttrVyVzPhiWp/ttrVyVzPhiWp_interp.pkl".format(self.task)


        self.fill_value = 1.0

        self.ttrVxVzThetaWt = None
        self.ttrVyVzPhiWp   = None

        self.ttrVxVzThetaWt_interp = None
        self.ttrVyVzPhiWp_interp   = None

        # Init matlab engine and workspace
        self.mateng = MatEng.start_matlab()
        self.mateng.workspace['ttrVxVzThetaWt_filepath'] = self.ttrVxVzThetaWt_filepath
        self.mateng.workspace['ttrVyVzPhiWp_filepath']   = self.ttrVyVzPhiWp_filepath

    def setup(self):
        if os.path.exists(self.ttrVxVzThetaWt_interp_filepath) and os.path.exists(self.ttrVyVzPhiWp_interp_filepath):
            print("Directly load interp objects ...")
            with open(self.ttrVxVzThetaWt_interp_filepath, 'rb') as f1, open(self.ttrVyVzPhiWp_interp_filepath, 'rb') as f2:
                self.ttrVxVzThetaWt_interp = pickle.load(f1)
                self.ttrVyVzPhiWp_interp   = pickle.load(f2)

        else:
            state_dim_num = len(self.state_step_num)
            l = []
            for size in self.state_step_num:
                x = np.empty((size), dtype = float)
                l.append(x)
            self.state_grid = np.array(l, dtype = object)

            for i in range(state_dim_num):
                self.state_grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.state_step_num[i])

            # setup ttrVxVzThetaWt and its interp function
            print("starting interp ttrVxVzThetaWt ...")
            self.mateng.eval("load(ttrVxVzThetaWt_filepath);", nargout=0)
            self.ttrVxVzThetaWt = self.mateng.workspace['phi']
            np_ttrVxVzThetaWt = np.asarray(self.ttrVxVzThetaWt)
            self.mateng.eval("clear phi;", nargout=0)
            self.ttrVxVzThetaWt_interp = RegularGridInterpolator((self.state_grid[0],
                                                                  self.state_grid[2],
                                                                  self.state_grid[3],
                                                                  self.state_grid[4]),
                                                                  np_ttrVxVzThetaWt,
                                                                  bounds_error=False,
                                                                  fill_value=self.fill_value)
            # setup ttrVyVzPhiWp and its interp function
            print("starting interp ttrVyVzPhiWp ...")
            self.mateng.eval("load(ttrVyVzPhiWp_filepath)", nargout=0)
            self.ttrVyVzPhiWp = self.mateng.workspace['phi']
            np_ttrVyVzPhiWp = np.asarray(self.ttrVyVzPhiWp)
            self.mateng.eval("clear phi;", nargout=0)
            self.ttrVyVzPhiWp_interp = RegularGridInterpolator((self.state_grid[1],
                                                                self.state_grid[2],
                                                                self.state_grid[5],
                                                                self.state_grid[6]),
                                                                np_ttrVyVzPhiWp,
                                                                bounds_error=False,
                                                                fill_value=self.fill_value)
            # save computed interp functions as objects
            with open(self.ttrVxVzThetaWt_interp_filepath, 'wb') as f1, \
                open(self.ttrVyVzPhiWp_interp_filepath, 'wb') as f2:
                pickle.dump(self.ttrVxVzThetaWt_interp, f1)
                pickle.dump(self.ttrVyVzPhiWp_interp, f2)
                print("saving interp objects ...")

    # full state s would be: [z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w]
    def interp(self, s):
        s = np.asarray(s)
        tmp_ttr = (self.ttrVxVzThetaWt_interp(s[[1, 3, 5, 8]]),
                   self.ttrVyVzPhiWp_interp(s[[2, 3, 4, 7]]))
        res = np.max(tmp_ttr, axis=0)
        return res[0]

    def vis_interp_res(self):

        vx_axis = np.linspace(-5, 5, 101)
        vz_axis = np.linspace(-5, 5, 101)
        theta_axis = np.linspace(-np.pi/2, np.pi/2, 18)
        wt_axis = np.linspace(-10 * np.pi, 10 * np.pi, 201)

        res = np.zeros((101, 101))
        print("starting visualizing interp ...")
        for ivx, vx in enumerate(vx_axis):
            for ivz, vz in enumerate(vz_axis):
                tmp_res = self.ttrVxVzThetaWt_interp([vx, vz, 0, 0])
                res[ivx, ivz] = tmp_res
        print("ending visualizing interp ...")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        CS = ax.contour(vx_axis, vz_axis, res[:, :])
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Simplest default with labels')
        plt.show()

def test():
    start_time = time.time()
    # your code
    th = ttr_helper()
    th.setup()
    setup_time = time.time()
    print("setup takes:", setup_time-start_time)
    for i in range(100):
        print(th.interp([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    forward_time = time.time()
    print("forward 100 times takes:", forward_time - setup_time)

    th.vis_interp_res()


if __name__ == "__main__":
    test()