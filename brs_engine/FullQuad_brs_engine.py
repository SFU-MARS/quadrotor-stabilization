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
#
# Subsystems:
#   VxTheta = ...
#   VzTheta = ...
#   VyPhi   = ...
#   VzPhi   = ...
#   Wt      = ... 
#   Wp      = ...



class FullQuad_brs_engine(object):
    
    def __init__(self):
        
        self.eng = matlab.engine.start_matlab()
        self.eng.cd("../brs_engine", nargout=0)
        self.eng.eval("addpath(genpath('../../toolboxls/Kernel'))", nargout=0)
        self.eng.eval("addpath(genpath('../../helperOC'))", nargout=0)

    def reset_variables(self, tMax=1.0, interval=0.1, nPoints=41):
        
        self.state_dim = 7
        self.ThrustMin = 0
        self.ThrustMax = 0.058 * 9.81 / 4

        self.goalCenter = matlab.double([[0], [0], [0], [0], [0], [0], [0]])
        self.goalRadius = matlab.double([[0.2], [0.2], [0.2], [np.pi/6], [np.pi/6], [0.2], [0.2]])
        self.goalLower = matlab.double([[-0.2], [-0.2], [-0.2], [-np.pi/6], [-np.pi/6], [-0.2], [-0.2]])
        self.goalUpper = matlab.double([[0.2], [0.2], [0.2], [np.pi/6], [np.pi/6], [0.2], [0.2]])
        
        # Note: pitch range in gazebo: [-pi/2, pi/2]
        # Note: roll range in gazebo: [-pi, pi]
        self.gMin = matlab.double([[-2.], [-2.], [-2.], [-np.pi/2], [-np.pi], [-np.pi/2], [-np.pi/2]])
        self.gMax = matlab.double([[2.], [2.], [2.], [np.pi/2], [np.pi], [np.pi/2], [np.pi/2]])
        self.gN   = matlab.double((nPoints * np.ones((self.state_dim, 1))).tolist())
        

        self.T1Min = self.T2Min = float(self.ThrustMin)
        self.T1Max = self.T2Max = float(self.ThrustMax)
        self.wtRange = self.wpRange = matlab.double([[-np.pi/2, np.pi/2]])
    
        self.tMax = float(tMax)
        self.interval = float(interval)
    
    def get_init_target(self):
       # cur_path = os.getcwd() 
       # if os.path.exists(cur_path + '/init_VxVzThetaWt.mat') and os.path.exists(cur_path + '/init_VyVzPhiWp.mat'):
       #     self.eng.eval("load init_VxVzThetaWt.mat init_VxVzThetaWt", nargout=0)
       #     self.eng.eval("load init_VyVzPhiWp.mat init_VyVzPhiWp", nargout=0)
       #     self.initTargetArea_VxVzThetaWt = self.eng.workspace['init_VxVzThetaWt']
       #     self.initTargetArea_VyVzPhiWp = self.eng.workspace['init_VyVzPhiWp']
       #     print("initial target created!")
       # else:
        (self.initTargetArea_VxTheta, self.initTargetArea_VzTheta, self.initTargetArea_VyPhi, self.initTargetArea_VzPhi, self.initTargetArea_Wt, self.initTargetArea_Wp) = \
            self.eng.Quad7D_create_init_target(self.gMin,
                                                self.gMax,
                                                self.gN,
                                                self.goalLower,
                                                self.goalUpper,
                                                self.goalCenter,
                                                self.goalRadius,
                                                nargout=6)
        self.eng.workspace['init_VxTheta'] = self.initTargetArea_VxTheta
        self.eng.workspace['init_VzTheta'] = self.initTargetArea_VzTheta
        self.eng.workspace['init_VyPhi']   = self.initTargetArea_VyPhi
        self.eng.workspace['init_VzPhi']   = self.initTargetArea_VzPhi
        self.eng.workspace['init_Wt'] = self.initTargetArea_Wt
        self.eng.workspace['init_Wp'] = self.initTargetArea_Wp

        self.eng.eval("save target/init_VxTheta.mat init_VxTheta", nargout=0)
        self.eng.eval("save target/init_VzTheta.mat init_VzTheta", nargout=0)
        self.eng.eval("save target/init_VyPhi.mat init_VyPhi", nargout=0)
        self.eng.eval("save target/init_VzPhi.mat init_VzPhi", nargout=0)
        self.eng.eval("save target/init_Wt.mat init_Wt", nargout=0)
        self.eng.eval("save target/init_Wp.mat init_Wp", nargout=0)

        print("initial target created and saved!")
        


    def get_value_function(self):
        (self.value_VxTheta, self.value_VzTheta, self.value_VyPhi, self.value_VzPhi, self.value_Wt, self.value_Wp) = \
            self.eng.Quad7D_calcu_RS(self.gMin,
                                      self.gMax,
                                      self.gN,
                                      self.T1Min,
                                      self.T1Max,
                                      self.T2Min,
                                      self.T2Max,
                                      self.wtRange,
                                      self.wpRange,
                                      self.initTargetArea_VxTheta,
                                      self.initTargetArea_VzTheta,
                                      self.initTargetArea_VyPhi,
                                      self.initTargetArea_VzPhi,
                                      self.initTargetArea_Wt,
                                      self.initTargetArea_Wp,
                                      self.tMax,
                                      self.interval,
                                      nargout=6)
        self.eng.workspace['value_VxTheta'] = self.value_VxTheta
        self.eng.workspace['value_VzTheta'] = self.value_VzTheta
        self.eng.workspace['value_VyPhi']   = self.value_VyPhi
        self.eng.workspace['value_VzPhi']   = self.value_VzPhi
        self.eng.workspace['value_Wt'] = self.value_Wt
        self.eng.workspace['value_Wp'] = self.value_Wp

        self.eng.eval("save value/value_VxTheta.mat value_VxTheta", nargout=0)
        self.eng.eval("save value/value_VzTheta.mat value_VzTheta", nargout=0)
        self.eng.eval("save value/value_VyPhi.mat value_VyPhi", nargout=0)
        self.eng.eval("save value/value_VzPhi.mat value_VzPhi", nargout=0)
        self.eng.eval("save value/value_Wt.mat value_Wt", nargout=0)
        self.eng.eval("save value/value_Wp.mat value_Wp", nargout=0)


        print("value function calculated and saved!")

    def get_ttr_function(self):
        (self.ttr_VxTheta, self.ttr_VzTheta, self.ttr_VyPhi, self.ttr_VzPhi, self.ttr_Wt, self.ttr_Wp) = \
            self.eng.Quad7D_calcu_TTR(self.gMin,
                                       self.gMax,
                                       self.gN,
                                       self.value_VxTheta,
                                       self.value_VzTheta,
                                       self.value_VyPhi,
                                       self.value_VzPhi,
                                       self.value_Wt,
                                       self.value_Wp,
                                       self.tMax,
                                       self.interval,
                                       nargout=6)

        self.eng.workspace['ttr_VxTheta'] = self.ttr_VxTheta
        self.eng.workspace['ttr_VzTheta'] = self.ttr_VzTheta
        self.eng.workspace['ttr_VyPhi']   = self.ttr_VyPhi
        self.eng.workspace['ttr_VzPhi']   = self.ttr_VzPhi
        self.eng.workspace['ttr_Wt'] = self.ttr_Wt
        self.eng.workspace['ttr_Wp'] = self.ttr_Wp

        self.eng.eval("save ttr/ttr_VxTheta.mat ttr_VxTheta", nargout=0)
        self.eng.eval("save ttr/ttr_VzTheta.mat ttr_VzTheta", nargout=0)
        self.eng.eval("save ttr/ttr_VyPhi.mat ttr_VyPhi", nargout=0)
        self.eng.eval("save ttr/ttr_VzPhi.mat ttr_VzPhi", nargout=0)
        self.eng.eval("save ttr/ttr_Wt.mat ttr_Wt", nargout=0)
        self.eng.eval("save ttr/ttr_Wp.mat ttr_Wp", nargout=0)


        print("ttr function calculated and saved!")
        

if __name__ == "__main__" :
    
    quadEngine =  FullQuad_brs_engine()
    quadEngine.reset_variables()
    quadEngine.get_init_target()
    quadEngine.get_value_function()
    quadEngine.get_ttr_function()










    
