classdef Quad7D_VyVzPhiWp < DynSys
    
    properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min

    % "real" parameters
    m % mass
    transDrag % translational drag
    rotDrag % rotational drag
    l % length of board
    Ixx % momentum inertia on x-axis
    grav % gravity
    end

methods
    function obj = Quad7D_VyVzPhiWp(x, T1Min, T1Max, T2Min, T2Max, m, ...
        transDrag, rotDrag, l, Ixx)
        % Here the dynamics follows the 'BaRC' paper since gazebo's routine for "roll" is same as what is         % in 'BaRC'
        %
        % Dynamics:
        % \dot v_y = -transDrag*v_y/m - T1*sin(\phi)/m - T2*sin(\phi)/m
        % \dot v_z = -(m*g + transDrag*v_z)/m + T1*cos(\phi)/m + T2*cos(\phi)/m
        % \dot \phi = \omega_p
        % \dot \omega_p = (-1/Ixx)*rotDrag*\omega - l*T1/Ixx + l*T2/Ixx
        %
        %
        % Input:
        % T1Max, T1Min, T2Max, T2Min
        % m - mass
        % grav - gravity
        % transDrag - transitional Drag 
        % rotDrag - rotational Drag
        % l - length

        if nargin < 2
            T1Max = 0.14;
        end
        
        if nargin < 3
            T1Min = 0;
        end

        if nargin < 4
            T2Max = 0.14;
        end

        if nargin < 5
            T2Min = 0;
        end
        
        if nargin < 6
            m = 0.027; % kg
        end
         
		if nargin < 7
            % from Julian Foster's paper
            transDrag = 9.17 * 10^-7; % kg*rad^-1
        end

        if nargin < 8
            rotDrag = 10.31 * 10^-7; % kg*rad^-1
        end

        if nargin < 9
            l = 0.039 %m
        end

        if nargin < 10
            Ixx = 1.6 * 10^-5 % kg*m^2
        end

        % Basic quad properties
        obj.nx = 4;
        obj.nu = 2;
        
        obj.x = x; % x: state
        obj.xhist = obj.x;
        
        obj.T1Max = T1Max;
        obj.T1Min = T1Min;
        obj.T2Max = T2Max;
        obj.T2Min = T2Min;
        obj.m = m;
        obj.transDrag = transDrag;
        obj.rotDrag = rotDrag;
        obj.l = l;
        obj.Ixx = Ixx;

        obj.grav = 9.81;
    end
end % end methods

end % end classdef










