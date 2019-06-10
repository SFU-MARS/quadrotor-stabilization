function [dataVxVzThetaW] = ...
  Quad4D_calcu_RS(gMin, gMax, gN, T1Min, T1Max, T2Min, T2Max, targetVxVzThetaW, tMax, interval)

% Directly compute the backward reachable set for the 4D quadrotor
%
%
% This function requires the helperOC and toolboxls libraries, found at
%     https://github.com/HJReachability/helperOC.git
%     https://bitbucket.org/ian_mitchell/toolboxls
%
% Inputs:
%   - (gMin, gMax, gN):
%         grid parameters
%   - (T1Min, T1Max, T2Min, T2Max):
%         system parameters
%   - tMax:
%         Time horizon of reachable set
%   - (targetVxVzThetaW):
%         target sets for system: (V_x, V_z, Theta, W)
%
% Outputs:
%   - (dataVxVzThetaW)
%         value functions for system: (V_x, V_z, Theta, W) 
% 
% Xubo Lyu, 2019-06-09

global gVxVzThetaW; % Gotta make this usable from other functions.

%% Target and obstacles
% 4D grid limits (v_x, v_z, theta, w)
if nargin < 3
  gMin = [-10;   -10;   0;      0     ];
  gMax = [10;    10;    2*pi;   2*pi  ];
  gN =   [81;    81;    101;    51    ];
end

% System dimensions
VxVzThetaW_dims = [1 2 3 4];


% Create grid structures for computation
% pass


%% Quadrotor parameters
% Constants
g = 9.81;   % gravity
m = 0.027;  % kg

% Crazyflie params
% maximum takeoff weight: 42g
% maximum thrust for four motors: 58g
% mass: 27g

if nargin < 4
  % thrust range
  T1Min = 0;
  T1Max = 0.14; % N
  T2Min = 0;
  T2Max = 0.14; % N

end

%% Time parameters
if nargin < 11
  tMax = 1.0;
end

 % Time horizon and intermediate results
tau = 0:interval:tMax;

%% Dynamical system
q_VxVzThetaW = Quad4D_VxVzThetaW([], T1Min, T1Max, T2Min, T2Max);

%% Compute reachable set
% Solver parameters
uMode = 'min';
vis = false;
quiet = true;
keepLast = false;

sDVxVzThetaW.dynSys = q_VxVzThetaW;
sDVxVzThetaW.grid = gVxVzThetaW;
sDVxVzThetaW.uMode = uMode;

eAVxVzThetaW.visualize = vis;
eAVxVzThetaW.quiet = quiet;
eAVxVzThetaW.keepLast = keepLast;


% Call solver

dataVxVzThetaW = HJIPDE_solve(targetVxVzThetaW, tau, sDVxVzThetaW, 'none', eAVxVzThetaW);

end
