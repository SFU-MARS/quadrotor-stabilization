function [targetVxTheta, targetVzTheta, targetVyPhi, targetVzPhi, targetWt, targetWp] = ...
    Quad7D_create_init_target(gMin, gMax, gN, goalLower, goalUpper, goalCenter, goalRadius)
% Xubo Lyu, 2019-06-09

global gVxTheta gVzTheta gVyPhi gVzPhi gWt gWp;

%% system dims
%% (1) v_x =
%% (2) v_y =
%% (3) v_z =
%% (4) \theta = 
%% (5) \phi =
%% (6) w_theta = 
%% (7) w_phi =

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

%% Initial target set
targetVxTheta = shapeRectangleByCorners(gVxTheta, goalLower(VxTheta_dims), goalUpper(VxTheta_dims));
targetVzTheta = shapeRectangleByCorners(gVzTheta, goalLower(VzTheta_dims), goalUpper(VzTheta_dims));
targetVyPhi   = shapeRectangleByCorners(gVyPhi, goalLower(VyPhi_dims), goalUpper(VyPhi_dims));
targetVzPhi   = shapeRectangleByCorners(gVzPhi, goalLower(VzPhi_dims), goalUpper(VzPhi_dims));
targetWt      = shapeCylinder(gWt, [], goalCenter(Wt_dims), goalRadius(Wt_dims));
targetWp      = shapeCylinder(gWp, [], goalCenter(Wp_dims), goalRadius(Wp_dims));


end
