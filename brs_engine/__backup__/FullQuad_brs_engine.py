import matlab.engine
import os
import numpy as np

from scipy.interpolate import RectBivariateSpline, interp1d


# Dynamics:
#   v_x = ...
#   v_y = ...
#   v_z = ...
#   \theta = ...
#   \phi = ...
#   w_t = ...
#   w_p = ...



class FullQuad_brs_engine(object):
    
    def __init__(self):
        
        self.eng = matlab.engine.start_matlab()
        self.eng.cd("../brs_engine", nargout=0)
        self.eng.eval("addpath(genpath('../../toolboxls/Kernel'))", nargout=0)
        self.eng.eval("addpath(genpath('../../helperOC'))", nargout=0)

    def reset_variables(self, tMax=0.5, interval=0.01, nPoints=41):
        
        self.state_dim = 7
        self.ThrustMin = 0
        self.ThrustMax = 0.058 * 9.81 / 4

        self.goalCenter = matlab.double([[0], [0], [0], [0], [0], [0], [0]])
        self.goalLower = matlab.double([[-0.2], [-0.2], [-0.2], [-np.pi/6], [-np.pi/6], [-0.2], [-0.2]])
        self.goalUpper = matlab.double([[0.2], [0.2], [0.2], [np.pi/6], [np.pi/6], [0.2], [0.2]])
        
        # Note: pitch range in gazebo: [-pi/2, pi/2]
        # Note: roll range in gazebo: [-pi, pi]
        self.gMin = matlab.double([[-2.], [-2.], [-2.], [-np.pi/2], [-np.pi], [-np.pi/2], [-np.pi/2]])
        self.gMax = matlab.double([[2.], [2.], [2.], [np.pi/2], [np.pi], [np.pi/2], [np.pi/2]])
        self.gN   = matlab.double((nPoints * np.ones((self.state_dim, 1))).tolist())
        

        self.T1Min = self.T2Min = float(self.ThrustMin)
        self.T1Max = self.T2Max = float(self.ThrustMax)


        self.tMax = float(tMax)
        self.interval = float(interval)
    
    def get_init_target(self):
        cur_path = os.getcwd() 
        if os.path.exists(cur_path + '/init_VxVzThetaWt.mat') and os.path.exists(cur_path + '/init_VyVzPhiWp.mat'):
            self.eng.eval("load init_VxVzThetaWt.mat init_VxVzThetaWt", nargout=0)
            self.eng.eval("load init_VyVzPhiWp.mat init_VyVzPhiWp", nargout=0)
            self.initTargetArea_VxVzThetaWt = self.eng.workspace['init_VxVzThetaWt']
            self.initTargetArea_VyVzPhiWp = self.eng.workspace['init_VyVzPhiWp']
            print("initial target created!")
        else:
            (self.initTargetArea_VxVzThetaWt, self.initTargetArea_VyVzPhiWp) = \
                self.eng.Quad7D_create_init_target(self.gMin,
                                                    self.gMax,
                                                    self.gN,
                                                    self.goalLower,
                                                    self.goalUpper,
                                                    nargout=2)
            self.eng.workspace['init_VxVzThetaWt'] = self.initTargetArea_VxVzThetaWt
            self.eng.workspace['init_VyVzPhiWp'] = self.initTargetArea_VyVzPhiWp
            self.eng.eval("save init_VxVzThetaWt.mat init_VxVzThetaWt", nargout=0)
            self.eng.eval("save init_VyVzPhiWp.mat init_VyVzPhiWp", nargout=0)
            print("initial target created!")
            


    def get_value_function(self):
        (self.valueVxVzThetaWt, self.valueVyVzPhiWp) = \
            self.eng.Quad7D_calcu_RS(self.gMin,
                                      self.gMax,
                                      self.gN,
                                      self.T1Min,
                                      self.T1Max,
                                      self.T2Min,
                                      self.T2Max,
                                      self.initTargetArea_VxVzThetaWt,
                                      self.initTargetArea_VyVzPhiWp,
                                      self.tMax,
                                      self.interval,
                                      nargout=2)
        self.eng.workspace['value_VxVzThetaWt'] = self.valueVxVzThetaWt
        self.eng.workspace['value_VyVzPhiWp'] = self.valueVyVzPhiWp
        self.eng.eval("save value_VxVzThetaWt.mat value_VxVzThetaWt", nargout=0)
        self.eng.eval("save value_VyVzPhiWp.mat value_VyVzPhiWp", nargout=0)
         
        print("value function calculated!")

    def get_ttr_function(self):
        (self.ttrVxVzThetaWt, self.ttrVyVzPhiWp) = \
            self.eng.Quad7D_calcu_TTR(self.gMin,
                                       self.gMax,
                                       self.gN,
                                       self.valueVxVzThetaWt,
                                       self.valueVyVzPhiWp,
                                       self.tMax,
                                       self.interval,
                                       nargout=2)

        self.eng.workspace['ttrVxVzThetaWt'] = self.ttrVxVzThetaWt
        self.eng.workspace['ttrVyVzPhiWp']   = self.ttrVyVzPhiWp
        
        self.eng.eval("save ttrVxVzThetaWt.mat ttrVxVzThetaWt", nargout=0)
        self.eng.eval("save ttrVyVzPhiWp.mat ttrVyVzPhiWp", nargout=0)

        print("ttr function calculated and saved!")
        

if __name__ == "__main__" :
    
    quadEngine =  FullQuad_brs_engine()
    quadEngine.reset_variables()
    quadEngine.get_init_target()
    quadEngine.get_value_function()
    quadEngine.get_ttr_function()










    
