function [dataVxVzThetaWt, dataVyVzPhiWp] = ...
  Quad7D_calcu_RS(gMin, gMax, gN, T1Min, T1Max, T2Min, T2Max, targetVxVzThetaWt, targetVyVzPhiWp, tMax, interval)

% Compute the backward reachable set for two 4D subsystems for 7D quadrotor full system
% 
%  v_x = ...
%  v_y = ...
%  v_z = ...
%  \theta = ...
%  \phi = ...
%  w_theta = ...
%  w_phi = ...
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
%   - (targetVxVzThetaWt):
%         target sets for subsystem: (V_x, V_z, Theta, Wt)
%
% Outputs:
%   - (dataVxVzThetaWt)
%         value functions for system: (V_x, V_z, Theta, Wt) 
% 
% Xubo Lyu, 2019-06-09

global gVxVzThetaWt  gVyVzPhiWp; % Gotta make this usable from other functions.

VxVzThetaWt_dims = [1 3 4 6];
VyVzPhiWp_dims = [2 3 5 7];

alreadyMade = sum(size(gVxVzThetaWt)) > 0;
if ~alreadyMade
    % Create grid structures for computation
    pdDims = 3; % \theta is periodic but \omega is not
    gVxVzThetaWt = createGrid(gMin(VxVzThetaWt_dims), gMax(VxVzThetaWt_dims), gN(VxVzThetaWt_dims), 3);
    gVyVzPhiWp   = createGrid(gMin(VyVzPhiWp_dims), gMax(VyVzPhiWp_dims), gN(VyVzPhiWp_dims), 3);
end


%% Quadrotor parameters
% Constants
g = 9.81;   % gravity
m = 0.027;  % kg

if nargin < 3
  error('you must set gMin, gMax and gN for computation!');
end
% Crazyflie params
% maximum takeoff weight: 42g
% maximum thrust for four motors: 58g
% mass: 27g

if nargin < 4
  % thrust range
  T1Min = 0;
  T1Max = 0.14; % Newton
  T2Min = 0;
  T2Max = 0.14; % Newton

end

%% Time parameters
if nargin < 11
  tMax = 1.0;
end

 % Time horizon and intermediate results
tau = 0:interval:tMax;

%% Dynamical subsystem
q_VxVzThetaWt = Quad7D_VxVzThetaWt([], T1Min, T1Max, T2Min, T2Max);
q_VyVzPhiWp   = Quad7D_VyVzPhiWp([], T1Min, T1Max, T2Min, T2Max);

%% Compute reachable set
% Solver parameters
uMode = 'min';
vis = false;
quiet = false;
keepLast = false;

% sD and eA for subsystem VxVzThetaWt
sDVxVzThetaWt.dynSys = q_VxVzThetaWt;
sDVxVzThetaWt.grid = gVxVzThetaWt;
sDVxVzThetaWt.uMode = uMode;

eAVxVzThetaWt.visualize = vis;
eAVxVzThetaWt.quiet = quiet;
eAVxVzThetaWt.keepLast = keepLast;

% sD and eA for subsystem VyVzPhiWp
sDVyVzPhiWp.dynSys = q_VyVzPhiWp;
sDVyVzPhiWp.grid = gVyVzPhiWp;
sDVyVzPhiWp.uMode = uMode;

eAVyVzPhiWp.visualize = vis;
eAVyVzPhiWp.quiet = quiet;
eAVyVzPhiWp.KeepLast = keepLast;


% Call solver
tic;
dataVxVzThetaWt = HJIPDE_solve(targetVxVzThetaWt, tau, sDVxVzThetaWt, 'none', eAVxVzThetaWt);
fprintf('HJE for VxVzThetaWt solved! \n');
t1=toc;

dataVyVzPhiWp   = HJIPDE_solve(targetVyVzPhiWp, tau, sDVyVzPhiWp, 'none', eAVyVzPhiWp);
fprintf('HJE for VyVzPhiWp solved! \n');
t2=toc;
end
