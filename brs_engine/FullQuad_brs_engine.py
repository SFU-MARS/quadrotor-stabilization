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

VX_IDX = 0
VY_IDX = 1
VZ_IDX = 2
THETA_IDX = 3
PHI_IDX = 4
WT_IDX = 5
WP_IDX = 6

class FullQuad_brs_engine(object):
    
    def __init__(self):
        
        self.eng = matlab.engine.start_matlab()
        home_path = os.environ['PROJ_HOME_2']
        self.eng.workspace['home_path'] = home_path
        
        self.eng.eval("addpath(genpath([home_path, '/brs_engine']));", nargout=0)
        self.eng.eval("addpath(genpath([home_path, '/toolboxls/Kernel']));", nargout=0)
        self.eng.eval("addpath(genpath([home_path, '/helperOC']));", nargout=0)
        
        self.reset_variables(tMax=10.0)    
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.eng.workspace['cur_path'] = cur_path
        
        if os.path.exists(cur_path + '/ttr/ttr_VxTheta.mat') and os.path.exists(cur_path + '/ttr/ttr_VzTheta.mat') and os.path.exists(cur_path + '/ttr/ttr_VyPhi.mat') \
            and os.path.exists(cur_path + '/ttr/ttr_VzPhi.mat') and os.path.exists(cur_path + '/ttr/ttr_Wt.mat') and os.path.exists(cur_path + '/ttr/ttr_Wp.mat'):
            print("asdfg")
            self.eng.eval("load([cur_path, '/ttr/ttr_VxTheta.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/ttr/ttr_VzTheta.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/ttr/ttr_VyPhi.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/ttr/ttr_VzPhi.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/ttr/ttr_Wt.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/ttr/ttr_Wp.mat']);", nargout=0)

            self.ttr_VxTheta = self.eng.workspace['ttr_VxTheta']
            self.ttr_VzTheta = self.eng.workspace['ttr_VzTheta']
            self.ttr_VyPhi   = self.eng.workspace['ttr_VyPhi']
            self.ttr_VzPhi   = self.eng.workspace['ttr_VzPhi']
            self.ttr_Wt      = self.eng.workspace['ttr_Wt']
            self.ttr_Wp      = self.eng.workspace['ttr_Wp']
        else:
            self.get_init_target()
            self.get_value_function()
            self.get_ttr_function()

        self.ttr_interpolation()

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
        
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.state_dim)]

        self.T1Min = self.T2Min = float(self.ThrustMin)
        self.T1Max = self.T2Max = float(self.ThrustMax)
        self.wtRange = self.wpRange = matlab.double([[-np.pi/2, np.pi/2]])
    
        self.tMax = float(tMax)
        self.interval = float(interval)
    
    def get_init_target(self):
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

        self.eng.eval("save([cur_path, '/target/init_VxTheta.mat'], 'init_VxTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/target/init_VzTheta.mat'], 'init_VzTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/target/init_VyPhi.mat'], 'init_VyPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/target/init_VzPhi.mat'], 'init_VzPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/target/init_Wt.mat'], 'init_Wt');", nargout=0)
        self.eng.eval("save([cur_path, '/target/init_Wp.mat'], 'init_Wp');", nargout=0)
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

        self.eng.eval("save([cur_path, '/value/value_VxTheta.mat'], 'value_VxTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/value/value_VzTheta.mat'], 'value_VzTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/value/value_VyPhi.mat'], 'value_VyPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/value/value_VzPhi.mat'], 'value_VzPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/value/value_Wt.mat'], 'value_Wt');", nargout=0)
        self.eng.eval("save([cur_path, '/value/value_Wp.mat'], 'value_Wp');", nargout=0)
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

        self.eng.eval("save([cur_path, '/ttr/ttr_VxTheta.mat'], 'ttr_VxTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/ttr/ttr_VzTheta.mat'], 'ttr_VzTheta');", nargout=0)
        self.eng.eval("save([cur_path, '/ttr/ttr_VyPhi.mat'], 'ttr_VyPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/ttr/ttr_VzPhi.mat'], 'ttr_VzPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/ttr/ttr_Wt.mat'], 'ttr_Wt');", nargout=0)
        self.eng.eval("save([cur_path, '/ttr/ttr_Wp.mat'], 'ttr_Wp');", nargout=0)


        print("ttr function calculated and saved!")
    
    def ttr_interpolation(self):
        np_tVxTheta = np.asarray(self.ttr_VxTheta)
        np_tVzTheta = np.asarray(self.ttr_VzTheta)
        np_tVyPhi   = np.asarray(self.ttr_VyPhi)
        np_tVzPhi   = np.asarray(self.ttr_VzPhi)
        np_tWt      = np.asarray(self.ttr_Wt)[:, -1]
        np_tWp      = np.asarray(self.ttr_Wp)[:, -1]
        print('np_tVxTheta shape is', np_tVxTheta.shape, flush=True)
        print('np_tVzTheta shape is', np_tVzTheta.shape, flush=True)
        print('np_tVyPhi shape is', np_tVyPhi.shape, flush=True)
        print('np_tVzPhi shape is', np_tVzPhi.shape, flush=True)
        print('np_tWt shape is', np_tWt.shape, flush=True)
        print('np_tWp shape is', np_tWp.shape, flush=True)

        # Here we interpolate based on discrete ttr function
        self.vxtheta_ttr_check = RectBivariateSpline(x=self.axis_coords[VX_IDX], y=self.axis_coords[THETA_IDX], z=np_tVxTheta, kx=1, ky=1)
        self.vztheta_ttr_check = RectBivariateSpline(x=self.axis_coords[VZ_IDX], y=self.axis_coords[THETA_IDX], z=np_tVzTheta, kx=1, ky=1)
        self.vyphi_ttr_check   = RectBivariateSpline(x=self.axis_coords[VY_IDX], y=self.axis_coords[PHI_IDX], z=np_tVyPhi, kx=1,ky=1)
        self.vzphi_ttr_check   = RectBivariateSpline(x=self.axis_coords[VZ_IDX], y=self.axis_coords[PHI_IDX], z=np_tVzPhi, kx=1,ky=1)
        self.wt_ttr_check      = interp1d(x=self.axis_coords[WT_IDX], y=np_tWt, fill_value='extrapolate', kind='nearest')
        self.wp_ttr_check      = interp1d(x=self.axis_coords[WP_IDX], y=np_tWp, fill_value='extrapolate', kind='nearest')

    def evaluate_ttr(self, states):
        states = np.array(states)
        if states.ndim == 1:
            states = np.reshape(states, (1,-1))

        vxtheta_ttr_checker = self.vxtheta_ttr_check(states[:, VX_IDX], states[:, THETA_IDX], grid=False)
        vztheta_ttr_checker = self.vztheta_ttr_check(states[:, VZ_IDX], states[:, THETA_IDX], grid=False)
        vyphi_ttr_checker   = self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False)
        vzphi_ttr_checker   = self.vzphi_ttr_check(states[:, VZ_IDX], states[:, PHI_IDX], grid=False)
        wt_ttr_checker      = self.wt_ttr_check(states[:, WT_IDX])
        wp_ttr_checker      = self.wp_ttr_check(states[:, WP_IDX])

        assert not np.isnan(vxtheta_ttr_checker)
        assert not np.isnan(vztheta_ttr_checker)
        assert not np.isnan(vyphi_ttr_checker)
        assert not np.isnan(vzphi_ttr_checker)
        assert not np.isnan(wt_ttr_checker)
        assert not np.isnan(wp_ttr_checker)

        tmp_ttr = (vxtheta_ttr_checker, vztheta_ttr_checker, vyphi_ttr_checker, vzphi_ttr_checker, wt_ttr_checker, wp_ttr_checker)
        res = np.max(tmp_ttr, axis=0)
        
        return res 

if __name__ == "__main__" :
    
    quadEngine =  FullQuad_brs_engine()

    print(quadEngine.evaluate_ttr([0, 0, 0, 0, 0, 0, 0]))









    
