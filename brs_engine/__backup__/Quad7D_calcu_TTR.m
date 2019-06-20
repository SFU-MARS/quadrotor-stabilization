function [ttrVxVzThetaWt, ttrVyVzPhiWp] = ...
  Quad7D_calcu_TTR(gMin, gMax, gN, valueVxVzThetaWt, valueVyVzPhiWp, tMax, interval)

    global gVxVzThetaWt gVyVzPhiWp;

    VxVzThetaWt_dims = [1 3 4 6];
    VyVzPhiWp_dims   = [2 3 5 7];

    % Create grid structures for computation
    gVxVzThetaWt = createGrid(gMin(VxVzThetaWt_dims), gMax(VxVzThetaWt_dims), gN(VxVzThetaWt_dims), 3); 
    gVyVzPhiWp   = createGrid(gMin(VyVzPhiWp_dims), gMax(VyVzPhiWp_dims), gN(VyVzPhiWp_dims), 3);


    % Time horizon and intermediate results
    tau = 0:interval:tMax;

    ttrVxVzThetaWt  = TD2TTR(gVxVzThetaWt, valueVxVzThetaWt, tau);
    ttrVyVzPhiWp    = TD2TTR(gVyVzPhiWp, valueVyVzPhiWp, tau);

end
