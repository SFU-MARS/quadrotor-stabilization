function [targetVxVzThetaW] = ...
  Quad6D_create_init_target(gMin, gMax, gN, goalCenter, goalRadii)
% Xubo Lyu, 2019-06-09

global gVxVzThetaW;

%% Target and obstacles
VxVzThetaW_dims = [1,2,3,4];

alreadyMade = sum(size(gVxVzThetaW)) > 0;
if ~alreadyMade
    % Create grid structures for computation 
    gVxVzThetaW = createGrid(gMin(VxVzThetaW_dims), gMax(VxVzThetaW_dims), gN(VxVzThetaW_dims), 3);

end

% set ignore dims
ignore_dims = [3, 4];

%% Initial target set
targetVxVzThetaW = shapeCylinder(gVxVzThetaW, ignore_dims, goalCenter([1, 2]), goalRadii([1])); % radius is a scalar


end
