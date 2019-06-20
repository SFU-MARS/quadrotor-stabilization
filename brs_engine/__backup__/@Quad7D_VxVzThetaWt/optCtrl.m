function uOpt = optCtrl(obj, ~, x, deriv, uMode)

%% Input processing
if nargin < 5
    uMode = 'max';
end

if ~iscell(deriv)
    deriv = num2cell(deriv);
end

%% Optimal control
uOpt = cell(obj.nu, 1);

det = cell(obj.nu, 1);

%% (1) Vx = ... 
%% (2) Vz = ... 
%% (3) theta = ... 
%% (4) w_t = ... 

%% Two det elements for two controls: T1, T2 in X-Z plane. Remember there are slight difference with BaRC due to the Gazebo's angle direction routine
det{1} = deriv{1}.*(1/obj.m).*sin(x{3}) + deriv{2}.*(1/obj.m).*cos(x{3}) + deriv{4}.*(obj.l/obj.Iyy);

det{2} = deriv{1}.*(1/obj.m).*sin(x{3}) + deriv{2}.*(1/obj.m).*cos(x{3}) + deriv{4}.*(-obj.l/obj.Iyy);

uMin = [obj.T1Min; obj.T2Min];
uMax = [obj.T1Max; obj.T2Max];

if strcmp(uMode, 'max')
    for i = 1:obj.nu
        uOpt{i} = (det{i} >= 0)*uMax(i) + (det{i} < 0)*uMin(i);
    end

elseif strcmp(uMode, 'min')
    for i = 1:obj.nu
        uOpt{i} = (det{i} >= 0)*uMin(i) + (det{i} < 0)*uMax(i);
    end
end

end


