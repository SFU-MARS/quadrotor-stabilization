classdef Quad7D_Wt < DynSys
  
  properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min
    
    I % Moment of inertia
    l % length of quadrotor
    rotDrag % rotational drag
  end
  
  methods
    function obj = Quad7D_Wt(x, T1Min, T1Max, T2Min, T2Max, I, l, rotDrag)
      % Dynamics:
      %    \dot \omega_t  = -rotDrag*\omega_t/I + l*u1/I - l*u2/I

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
        I = 1.6 * 10^-5; 
      end
      
      if nargin < 7
        l = 0.065; %m
      end
      
      if nargin < 8
        rotDrag = 7.93 * 10^-12;
      end      
      
      % Basic vehicle properties
      obj.nx = 1;
      obj.nu = 2;  
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.T1Max = T1Max;
      obj.T1Min = T1Min;
      obj.T2Max = T2Max;
      obj.T2Min = T2Min;
      obj.I = I;
      obj.l = l;
      obj.rotDrag = rotDrag;

    end
    
  end % end methods
end % end classdef
