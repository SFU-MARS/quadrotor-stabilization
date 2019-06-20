classdef Quad7D_VzPhi < DynSys
  
  properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min
    
    % "Real" parameters
    m % mass
    
    grav % gravity
    
    transDrag %translational drag
    
    % Ficticious parameter for decomposition
    wpRange
  end
  
  methods
    function obj = Quad7D_VzPhi(x, T1Min, T1Max, T2Min, T2Max, wpRange, m, transDrag)
      % Constructor. Creates a quadcopter object with a unique ID,
      % state x, and reachable set information reachInfo
      %
      % Dynamics:
      %    \dot v_z  = -g - transDrag*v_z/m + T1*cos(\phi)/m + T2*cos(\phi)/m
      %    \dot \phi = \omega_p
      %
      % Inputs:
      %   T1Max, T1Min, T2Max, T2Min - limits on T1 and T2 (controls
      %   m - mass
      %   grav - gravity
      %   transDrag - translational Drag
      
      if nargin < 2
        T1Min = 0;
      end
      
      if nargin < 3
        T1Max = 0.14;
      end
      
      if nargin < 4
        T2Min = 0;
      end
      
      if nargin < 5
        T2Max = 0.14;
      end
      
      if nargin < 6
        wpRange = [0 2*pi];
      end      
       
      if nargin < 7
        m = 0.027;
      end

      if nargin < 8
        transDrag = 7.93 * 10^-12; 
      end
     

      % Basic vehicle properties
      obj.nx = 2;
      obj.nu = 3;  
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.T1Max = T1Max;
      obj.T1Min = T1Min;
      obj.T2Max = T2Max;
      obj.T2Min = T2Min;
      obj.m = m;
      obj.transDrag = transDrag;
      obj.wpRange = wpRange;
      
      obj.grav = 9.81;
    end
    
  end % end methods
end % end classdef
