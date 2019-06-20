function [dataVxTheta, dataVzTheta, dataVyPhi, dataVzPhi, dataWt, dataWp] = ...
  Quad7D_calcu_RS(gMin, gMax, gN, T1Min, T1Max, T2Min, T2Max, wtRange, wpRange, targetVxTheta, targetVzTheta, targetVyPhi, targetVzPhi, targetWt, targetWp, tMax, interval)

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

global gVxTheta gVzTheta gVyPhi gVzPhi gWt gWp;

VxTheta_dims = [1 4];
VzTheta_dims = [3 4];
VyPhi_dims   = [2 5];
VzPhi_dims   = [3 5];
Wt_dims = [6];
Wp_dims = [7];

alreadyMade = sum(size(gVxTheta)) > 0;
if ~alreadyMade
    % Create grid structures for computation
    gVxTheta = createGrid(gMin(VxTheta_dims), gMax(VxTheta_dims), gN(VxTheta_dims), 2);
    gVzTheta = createGrid(gMin(VzTheta_dims), gMax(VzTheta_dims), gN(VzTheta_dims), 2); 
    gVyPhi   = createGrid(gMin(VyPhi_dims), gMax(VyPhi_dims), gN(VyPhi_dims), 2);
    gVzPhi   = createGrid(gMin(VzPhi_dims), gMax(VzPhi_dims), gN(VzPhi_dims), 2);
    gWt      = createGrid(gMin(Wt_dims), gMax(Wt_dims), gN(Wt_dims));
    gWp      = createGrid(gMin(Wp_dims), gMax(Wp_dims), gN(Wp_dims));
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

  wtRange = [0 2*pi];
  wpRange = [0 2*pi];

end

%% Time parameters
if nargin < 16
  tMax = 1.0;
end

if nargin < 17
  interval = 0.1;
end

 % Time horizon and intermediate results
tau = 0:interval:tMax;

%% Dynamical subsystem
q_VxTheta = Quad7D_VxTheta([], T1Min, T1Max, T2Min, T2Max, wtRange);
q_VzTheta = Quad7D_VzTheta([], T1Min, T1Max, T2Min, T2Max, wtRange);
q_VyPhi   = Quad7D_VyPhi([], T1Min, T1Max, T2Min, T2Max, wpRange);
q_VzPhi   = Quad7D_VzPhi([], T1Min, T1Max, T2Min, T2Max, wpRange);
q_Wt      = Quad7D_Wt([], T1Min, T1Max, T2Min, T2Max);
q_Wp      = Quad7D_Wp([], T1Min, T1Max, T2Min, T2Max);

%% Compute reachable set
% Solver parameters
uMode = 'min';
vis = false;
quiet = false;
keepLast = false;

% sD and eA for subsystem VxTheta
sDVxTheta.dynSys = q_VxTheta;
sDVxTheta.grid = gVxTheta;
sDVxTheta.uMode = uMode;

eAVxTheta.visualize = vis;
eAVxTheta.quiet = quiet;
eAVxTheta.keepLast = keepLast;

% sD and eA for subsystem VzTheta
sDVzTheta.dynSys = q_VzTheta;
sDVzTheta.grid = gVzTheta;
sDVzTheta.uMode = uMode;

eAVzTheta.visualize = vis;
eAVzTheta.quiet = quiet;
eAVzTheta.KeepLast = keepLast;

% sD and eA for subsystem VyPhi
sDVyPhi.dynSys = q_VyPhi;
sDVyPhi.grid   = gVyPhi;
sDVyPhi.uMode  = uMode;

eAVyPhi.visualize = vis;
eAVyPhi.quiet = quiet;
eAVyPhi.keepLast = keepLast;

% sD and eA for subsystem VzPhi
sDVzPhi.dynSys = q_VzPhi;
sDVzPhi.grid   = gVzPhi;
sDVzPhi.uMode  = uMode;

eAVzPhi.visualize = vis;
eAVzPhi.quiet  = quiet;
eAVzPhi.keepLast = keepLast;

% sD and eA for subsystem Wt
sDWt.dynSys = q_Wt;
sDWt.grid = gWt; 
sDWt.uMode = uMode;

eAWt.visualize = vis;
eAWt.quiet  = quiet;
eAWt.keepLast = keepLast;

% sD and eA for subsystem Wp
sDWp.dynSys = q_Wp;
sDWp.grid = gWp;
sDWp.uMode = uMode;

eAWp.visualize = vis;
eAWp.quiet  = quiet;
eAWp.keepLast = keepLast;


% Call solver
tic;
dataVxTheta = HJIPDE_solve(targetVxTheta, tau, sDVxTheta, 'none', eAVxTheta);
fprintf('HJE for VxTheta solved! \n');
t1=toc;

dataVzTheta = HJIPDE_solve(targetVzTheta, tau, sDVzTheta, 'none', eAVzTheta);
fprintf('HJE for VzTheta solved! \n');
t2=toc;

dataVyPhi   = HJIPDE_solve(targetVyPhi, tau, sDVyPhi, 'none', eAVyPhi);
fprintf('HJE for VyPhi solved! \n');
t3=toc;


dataVzPhi   = HJIPDE_solve(targetVzPhi, tau, sDVzPhi, 'none', eAVzPhi);
fprintf('HJE for VzPhi solved! \n');
t4=toc;


dataWt      = HJIPDE_solve(targetWt, tau, sDWt, 'none', eAWt);
fprintf('HJE for Wt solved! \n');
t5=toc;


dataWp      = HJIPDE_solve(targetWp, tau, sDWp, 'none', eAWp);
fprintf('HJE for Wp solved! \n');
t6=toc;




end







