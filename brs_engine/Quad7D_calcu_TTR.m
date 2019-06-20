function [ttrVxTheta, ttrVzTheta, ttrVyPhi, ttrVzPhi, ttrWt, ttrWp] = ...
  Quad7D_calcu_TTR(gMin, gMax, gN, valueVxTheta, valueVzTheta, valueVyPhi, valueVzPhi, valueWt, valueWp, tMax, interval)

    global gVxTheta gVzTheta gVyPhi gVzPhi gWt gWp;

    VxTheta_dims = [1 4];
    VzTheta_dims = [3 4];
    VyPhi_dims   = [2 5];
    VzPhi_dims   = [3 5];
    Wt_dims = 6;
    Wp_dims = 7;

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



    % Time horizon and intermediate results
    tau = 0:interval:tMax;
    
    fprintf("computing ttr for system VxTheta \n");
    ttrVxTheta  = TD2TTR(gVxTheta, valueVxTheta, tau);

    fprintf("computing ttr for system VzTheta \n");
    ttrVzTheta  = TD2TTR(gVzTheta, valueVzTheta, tau);

    fprintf("computing ttr for system VyPhi \n");
    ttrVyPhi    = TD2TTR(gVyPhi, valueVyPhi, tau);

    fprintf("computing ttr for system VzPhi \n");
    ttrVzPhi    = TD2TTR(gVzPhi, valueVzPhi, tau);

    fprintf("computing ttr for system Wt \n");
    fprintf("valueWt size:");
    disp(size(valueWt));
    ttrWt       = TD2TTR(gWt, valueWt, tau);
    
    fprintf("valueWp size:");
    disp(size(valueWp));
    fprintf("computing ttr for system Wp \n");
    ttrWp       = TD2TTR(gWp, valueWp, tau);

end
