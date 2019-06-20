classdef Quad7D_VxVzThetaWt < DynSys
    
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
    Iyy % momentum inertia on y-axis
    grav % gravity 
    end

methods
    function obj = Quad7D_VxVzThetaWt(x, T1Min, T1Max, T2Min, T2Max, m, ...
        transDrag, rotDrag, l, Iyy)

        % Dynamics:
        % \dot v_x = -transDrag*v_x/m + T1*sin(\theta)/m + T2*sin(\theta)/m
        % \dot v_z = -(m*g + transDrag*v_z)/m + T1*cos(\theta)/m + T2*cos(\theta)/m 
        % \dot \theta = \omega_t
        % \dot \omega_t = -rotDrag*\omega_t/Iyy + l*T1/Iyy - l*T2/Iyy
        
        % Input:
        % T1Max, T1Min, T2Max, T2Min
        % m - mass
        % grav - gravity
        % transDrag - transitional Drag
        % rotDrag - rotational Drag
        % l - length

        if nargin < 2
            % crazyflie 42g max takeoff weight
            % crazyflie 58g max thrust in total
            T1Max = 0.14; % 0.058 * 9.81 / 4
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
            % transDrag = 9.17 * 10^-7; % kg*rad^-1
            transDrag = 7.93 * 10^-12;
        end

        if nargin < 8
            % rotDrag = 10.31 * 10^-7; % kg*rad^-1
            rotDrag = 7.93 * 10^-12;
        end

        if nargin < 9
            l = 0.039 %m
        end

        if nargin < 10
            Iyy = 1.6 * 10^-5 % kg*m^2
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
        obj.Iyy = Iyy;

        obj.grav = 9.81;
    end
end % end methods

end % end classdef


