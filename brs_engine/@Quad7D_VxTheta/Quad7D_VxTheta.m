classdef Quad7D_VxTheta < DynSys
    
    properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min

    % "real" parameters
    m % mass
    transDrag % translational drag
    % rotDrag % rotational drag
    % l % length of board
    % Iyy % momentum inertia on y-axis
    wtRange
    grav % gravity 
    end

methods
    function obj = Quad7D_VxTheta(x, T1Min, T1Max, T2Min, T2Max, wtRange, m, transDrag)

        % Dynamics:
        % \dot v_x = -transDrag*v_x/m + T1*sin(\theta)/m + T2*sin(\theta)/m
        % \dot \theta = \omega_t
        
        % Input:
        % T1Max, T1Min, T2Max, T2Min
        % m - mass
        % grav - gravity
        % transDrag - transitional Drag

        if nargin < 2
            % crazyflie 42g max takeoff weight
            % crazyflie 58g max thrust in total
            T1Min = 0; % 0.058 * 9.81 / 4
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
            wtRange = [0 2*pi];
        end

        if nargin < 7
            m = 0.027; % kg 
        end

        if nargin < 8
            % from Julian Foster's paper
            % transDrag = 9.17 * 10^-7; % kg*rad^-1
            transDrag = 7.93 * 10^-12;
        end

        % Basic quad properties
        obj.nx = 2;
        obj.nu = 3;
        
        obj.x = x; % x: state
        obj.xhist = obj.x;
        
        obj.T1Max = T1Max;
        obj.T1Min = T1Min;
        obj.T2Max = T2Max;
        obj.T2Min = T2Min;
        obj.m = m;
        obj.transDrag = transDrag;
        obj.wtRange = wtRange;

        obj.grav = 9.81;
    end
end % end methods

end % end classdef


