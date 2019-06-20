function [targetVxVzThetaWt, targetVyVzPhiWp] = ...
    Quad7D_create_init_target(gMin, gMax, gN, goalLower, goalUpper)
% Xubo Lyu, 2019-06-09

global gVxVzThetaWt  gVyVzPhiWp;

%% system dims
%% (1) v_x =
%% (2) v_y =
%% (3) v_z =
%% (4) \theta = 
%% (5) \phi =
%% (6) w_theta = 
%% (7) w_phi =

VxVzThetaWt_dims = [1 3 4 6];
VyVzPhiWp_dims = [2 3 5 7];

alreadyMade = sum(size(gVxVzThetaWt)) > 0;
if ~alreadyMade
    % Create grid structures for computation
    pdDims = 3; % \theta is periodic but \omega is not
    gVxVzThetaWt = createGrid(gMin(VxVzThetaWt_dims), gMax(VxVzThetaWt_dims), gN(VxVzThetaWt_dims), 3);
    gVyVzPhiWp   = createGrid(gMin(VyVzPhiWp_dims), gMax(VyVzPhiWp_dims), gN(VyVzPhiWp_dims), 3);
end

%% Initial target set
targetVxVzThetaWt = shapeRectangleByCorners(gVxVzThetaWt, goalLower(VxVzThetaWt_dims), goalUpper(VxVzThetaWt_dims));
targetVyVzPhiWp = shapeRectangleByCorners(gVyVzPhiWp, goalLower(VyVzPhiWp_dims), goalUpper(VyVzPhiWp_dims));


end
