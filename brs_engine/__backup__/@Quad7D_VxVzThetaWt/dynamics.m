function dx = dynamics(obj, ~, x, u, ~)
        % Dynamics:
        % \dot v_x = -transDrag*v_x/m + T1*sin(\theta)/m + T2*sin(\theta)/m
        % \dot v_z = -(m*g + transDrag*v_z)/m + T1*cos(\theta)/m + T2*cos(\theta)/m 
        % \dot \theta = \omega_t
        % \dot \omega_t = -Iyy*rotDrag*\omega_t + l*T1/Iyy - l*T2/Iyy
        
dx = cell(obj.nx, 1);

returnVector = false;
if ~iscell(x)
    returnVector = true
    x = num2cell(x);
    u = num2cell(u);
end

for dim = 1:obj.nx
    dx{dim} = dynamics_cell_helper(obj, x, u, dim);
end

if returnVector
    dx = cell2mat(dx);
end
end

function dx = dynamics_cell_helper(obj, x, u, dim)

switch dim
    case 1
        dx = (-obj.transDrag * x{1} / obj.m) + ...
            (sin(x{3}) .* u{1} / obj.m) + ...
            (sin(x{3}) .* u{2} / obj.m);
    case 2
        dx = (-(obj.m * obj.grav + obj.transDrag * x{2}) / obj.m) + ...
            (cos(x{3}) .* u{1} / obj.m) + ...
            (cos(x{3}) .* u{2} / obj.m);

    case 3
        dx = x{4};

    case 4
        dx = (-obj.rotDrag * x{4} / obj.Iyy) + ...
            (obj.l * u{1} / obj.Iyy) - ...
            (obj.l * u{2} / obj.Iyy);
    otherwise
        error('exceeding maximum dims of state!')
end
end





